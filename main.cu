#include <SFML/Graphics.hpp>
#include <cmath>
#include <bits/stdc++.h>
#include <iostream>
#include <stdio.h>
#include "./include/ACO.hpp"
#include "./include/GUI.hpp"
#include "./include/TSPLIB.hpp"

using namespace std;

__global__ void test_kernel(int N, int * counter)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	
	for(int i = idx; i < N; i+=stride)
	{
		printf("Kernel number %d, adding to counter (currently %d)\n", i, *counter);
		atomicAdd(counter, 1);
	}
}


// TODO: Precalc desire (currently takes two pow() calls and a multiplication
__global__ void run_one_ant(int NUMBER_OF_ANTS, float time, int dimension, float * pheromones, float * pheromones_delta, float * edge_weights, float a, float b, float p, float Q)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;

	
	for(int i = idx; i < NUMBER_OF_ANTS; i+=stride)
	{

		int number_of_remaining_nodes = dimension;
		int * remaining_nodes = (int*)malloc(sizeof(int)*dimension);
		for(int j=0;j<dimension;j++)
		{
			remaining_nodes[j]=j;
		}

		printf("Edge i to n-i = %.2f\n",edge_weights[i+(dimension-i-1)*dimension]);

		
		printf("Hello ant %d\n", i);
	}
}


float get_time()
{
	struct timespec time;
	timespec_get(&time, TIME_UTC);
	return time.tv_sec + time.tv_nsec/1000000000;
}


int main(int argc, char ** argv)
{
	if(argc < 2) {
		cout << "Please provide the filename of the .tsp file as an argument\n";
		exit(1);
	}

	float initial_time = get_time();

	string tsp_filename(argv[1]);

	sf::RenderWindow window( sf::VideoMode(1980,1080), "TSPLIB Display");

	sf::CircleShape circle_shape(20);

	circle_shape.setPosition(50,100);
	circle_shape.setOrigin(circle_shape.getRadius(), circle_shape.getRadius());
	circle_shape.setFillColor(sf::Color::Red);

	TSPLIB_INSTANCE test_tsp(tsp_filename);

	GUI gui(window);

	//set_view_edges_with_tsplib_instance(window, test_tsp);
	gui.set_camera_view_edges_given_points(test_tsp.dimension, test_tsp.node_coords);


	float * tsp_edge_weights = (float*)malloc(sizeof(float)*test_tsp.dimension * test_tsp.dimension);
	for(int i=0;i<test_tsp.dimension; i++)
	{
		for(int j=0;j<test_tsp.dimension; j++)
		{
			tsp_edge_weights[i+j*test_tsp.dimension] = test_tsp.euclidean_distance(i,j);
		}

	}

	int number_of_ants = 1000;
	ACO test_aco(test_tsp.dimension, tsp_edge_weights, 1, 5, 0.5, 1);

	float best_tour_length = numeric_limits<float>::max();
	int * best_tour = (int*)malloc(sizeof(int)*test_tsp.dimension);
	for(int i=0;i<test_tsp.dimension;i++)
	{
		best_tour[i]=i;
	}

	int current_iteration = 0;

	int * cuda_counter;
	cudaMallocManaged(&cuda_counter, sizeof(int));

	float * pheromones;
	float * pheromones_delta;
	float * edge_weights;
	cudaMallocManaged(&pheromones, sizeof(float)*test_tsp.dimension*test_tsp.dimension);
	cudaMallocManaged(&pheromones_delta, sizeof(float)*test_tsp.dimension*test_tsp.dimension);
	cudaMallocManaged(&edge_weights, sizeof(float)*test_tsp.dimension*test_tsp.dimension);
	cudaMemcpy(pheromones, test_aco.pheromones,sizeof(float)*test_tsp.dimension*test_tsp.dimension,::cudaMemcpyHostToDevice);
	cudaMemcpy(pheromones_delta, test_aco.pheromones_delta,sizeof(float)*test_tsp.dimension*test_tsp.dimension,::cudaMemcpyHostToDevice);
	cudaMemcpy(edge_weights, tsp_edge_weights,sizeof(float)*test_tsp.dimension*test_tsp.dimension,::cudaMemcpyHostToDevice);
	*cuda_counter = 0;
	//test_kernel<<<10,10>>>(100,cuda_counter);
	cout << "Cuda counter = " << *cuda_counter << "\n";
	run_one_ant<<<10,10>>>(100, test_tsp.dimension, pheromones, pheromones_delta, edge_weights, test_aco.a, test_aco.b, test_aco.p, test_aco.Q);
	cudaDeviceSynchronize();
	exit(1);

	
	while (window.isOpen()) 
	{
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type ==
			sf::Event::Closed)
				window.close();

			// catch the resize events
			if (event.type == sf::Event::Resized)
			{
				// update the view to the new size of the window
				sf::FloatRect visibleArea(0, 0, event.size.width, event.size.height);
				window.setView(sf::View(visibleArea));
			}
		}

		window.clear(sf::Color::White);
		//window.draw(circle_shape);
		//drawThickLine(window, sf::Vector2f(100, 150), sf::Vector2f(250, 250), 10.0f, sf::Color::Red);
		float window_area_scale_factor = min(window.getSize().x, window.getSize().y) / 1000.0;
		//test_tsp.draw(window, 7*window_area_scale_factor, sf::Color::Blue);

		gui.draw_pheromones(test_tsp.dimension, test_tsp.node_coords, test_aco.pheromones, 20, sf::Color::Red);
		gui.draw_tour(test_tsp.dimension, test_tsp.node_coords, best_tour, 20, sf::Color::Black);
		gui.draw_points(test_tsp.node_coords, 15, sf::Color::Blue);
		window.display();

		for(int i=0;i<number_of_ants;i++){
			test_aco.run_one_ant();
			//cout << "Iteration " << i << "\n";
			//cout << "Tour Length = " << test_aco.get_length_of_tour() << ", ";
			if(test_aco.tour_length < best_tour_length)
			{ 
				best_tour_length = test_aco.tour_length;
				memcpy(best_tour, test_aco.tour, sizeof(int)*test_tsp.dimension);
				cout << "New best tour: " << best_tour_length << ", iteration = " << current_iteration << "\n";
				float current_time = get_time();
				cout << "Time: " << (int)(initial_time - current_time) << " seconds " << (((int)(initial_time - current_time)*1000)%1000 )<< " miliseconds\n";
				//for(int j=0;j<test_tsp.dimension;j++)
				//{
				//	cout << best_tour[j] << ((j==test_tsp.dimension-1)? "\n" : ", ");
				//}
			}
			//cout << "Best Tour Length = " << best_tour_length << "\n";
		}
		test_aco.end_epoch();
		current_iteration++;

	}
	return 0;
}

