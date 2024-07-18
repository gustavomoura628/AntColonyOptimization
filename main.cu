#include <SFML/Graphics.hpp>
#include <cmath>
#include <bits/stdc++.h>
#include <iostream>
#include <stdio.h>
#include "./include/ACO.hpp"
#include "./include/GUI.hpp"
#include "./include/TSPLIB.hpp"

#include <curand_kernel.h>


using namespace std;

__global__ void initialize_memory(int NUMBER_OF_ANTS, int dimension, float * pheromones, float * pheromones_delta, int * tour_preallocated_memory)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for(int i=tid;i<dimension*dimension;i+=stride)
	{
		pheromones[i] = 1;
		pheromones_delta[i] = 0;
	}

	for(int ant_index = tid; ant_index < NUMBER_OF_ANTS; ant_index+=stride)
	{
		for(int i=0;i<dimension;i++)
		{
			tour_preallocated_memory[i + ant_index*dimension] = i;
		}
	}
}

__global__ void finish_epoch(int NUMBER_OF_ANTS, int dimension, float * pheromones, float * pheromones_delta, float * edge_weights, int * tour_preallocated_memory, float p)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;
	for(int i=tid;i<dimension*dimension;i+=stride)
	{
		pheromones[i] = (1-p)*pheromones[i] + pheromones_delta[i];
		pheromones_delta[i] = 0;
	}
}


// TODO: Precalc desire (currently takes two pow() calls and a multiplication
// IDEA: Create tour automatically, instead of adding known cities to the end of remaining_cities, move them to the start.
__global__ void run_one_ant(int NUMBER_OF_ANTS, double time, int dimension, float * pheromones, float * pheromones_delta, float * edge_weights, float a, float b, float p, float Q, int * tour_preallocated_memory, float * tour_length)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x * gridDim.x;

	curandState rgnState;
	curand_init((unsigned long long)time, tid, 0, &rgnState);
	
	for(int ant_index = tid; ant_index < NUMBER_OF_ANTS; ant_index+=stride)
	{
		int number_of_nodes_traversed = 0;
		int * tour = &tour_preallocated_memory[ant_index*dimension];

		// Reseting the tour is not necessary, if you just initialize it once, the scrambled nature of a finished tour does not interfere with
		// the calculations, it just needs to contain all node indices.


		int current_city_index = (int)(curand_uniform(&rgnState)*(dimension-1)+0.5);
		int current_city = tour[current_city_index];

		while(number_of_nodes_traversed < dimension)
		{
			// swap();
			int temp = tour[current_city_index];
			tour[current_city_index] = tour[number_of_nodes_traversed];
			tour[number_of_nodes_traversed] = temp;

			number_of_nodes_traversed++;

			float total_desire = 0;
			for(int i=number_of_nodes_traversed;i<dimension;i++)
			{
				total_desire += pow(pheromones[current_city + tour[i]*dimension],a) * pow(1/edge_weights[current_city + tour[i]*dimension],b);
			}


			float random_number = curand_uniform(&rgnState);
			float probability_sum = 0;
			int i = number_of_nodes_traversed;
			//ACO algorithm
			int destination_city = tour[dimension-1]; // In case the random_number is exactly 1.0f
			int destination_city_index = dimension-1;
			for(;i<dimension;i++)
			{
				float desire = pow(pheromones[current_city + tour[i]*dimension],a) * pow(1/edge_weights[current_city + tour[i]*dimension],b);
				float probability = desire/total_desire;
				//printf("probability of going from city %d to %d = %.2f\n",current_city, i, probability);
				probability_sum += probability;

				if(probability_sum > random_number)
				{
					destination_city = tour[i];
					destination_city_index = i;
					break;
				}
			}
			current_city = destination_city;
			current_city_index = destination_city_index;
		}
		//printf("Current city index = %d\n",current_city_index);
		//printf("Hello, here is a random number: %.2f, ant %d sends its regards...\n", random_number, i);

		// calculate total length of tour
		float length = 0;
		for(int i=0;i<dimension;i++)
		{
			int city_i = tour[i];
			int city_j = tour[(i+1)%dimension];
			//printf("Weight from %d to %d = %.2f\n", city_i, city_j, edge_weights[city_i + city_j*dimension]);
			length += edge_weights[city_i + city_j*dimension];
		}
		//printf("Tour length = %.2f\n",length);
		tour_length[ant_index] = length;

		// increment pheromones_delta
		for(int i=0;i<dimension;i++)
		{
			int city_i = tour[i];
			int city_j = tour[(i+1)%dimension];
			atomicAdd(&pheromones_delta[city_i + city_j*dimension],Q/length);
		}
	}
}

double get_time()
{
	struct timespec time;
	timespec_get(&time, TIME_UTC);
	//cout << "get time = " << time.tv_sec << " seconds, " << time.tv_nsec << " nanoseconds\n";
	return (double)time.tv_sec + (double)time.tv_nsec/1000000000.0f;
}


int main(int argc, char ** argv)
{
	if(argc < 2) {
		cout << "Please provide the filename of the .tsp file as an argument\n";
		exit(1);
	}

	double initial_time = get_time();

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

	float * pheromones;
	float * pheromones_delta;
	float * edge_weights;
	int * tour_preallocated_memory;
	float * tour_length;
	cudaMallocManaged(&pheromones, sizeof(float)*test_tsp.dimension*test_tsp.dimension);
	cudaMallocManaged(&pheromones_delta, sizeof(float)*test_tsp.dimension*test_tsp.dimension);
	cudaMallocManaged(&edge_weights, sizeof(float)*test_tsp.dimension*test_tsp.dimension);
	cudaMallocManaged(&tour_preallocated_memory, sizeof(int)*test_tsp.dimension*number_of_ants);
	cudaMallocManaged(&tour_length, sizeof(float)*number_of_ants);

	cudaMemcpy(edge_weights, tsp_edge_weights, sizeof(float)*test_tsp.dimension*test_tsp.dimension,::cudaMemcpyHostToDevice);

	int deviceId;
	cudaGetDevice(&deviceId);
	cudaMemPrefetchAsync(pheromones, sizeof(float)*test_tsp.dimension*test_tsp.dimension, deviceId);
	cudaMemPrefetchAsync(pheromones_delta, sizeof(float)*test_tsp.dimension*test_tsp.dimension, deviceId);
	cudaMemPrefetchAsync(edge_weights, sizeof(float)*test_tsp.dimension*test_tsp.dimension, deviceId);
	cudaMemPrefetchAsync(tour_preallocated_memory, sizeof(int)*test_tsp.dimension*number_of_ants, deviceId);
	cudaMemPrefetchAsync(tour_length, sizeof(float)*number_of_ants, deviceId);



	int threads_per_block = 64;
	int number_of_blocks = number_of_ants/threads_per_block+1;

	initialize_memory<<<number_of_blocks,threads_per_block>>>(number_of_ants, test_tsp.dimension, pheromones, pheromones_delta, tour_preallocated_memory);

	cudaDeviceSynchronize();

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


		// PARALLEL

		gui.draw_pheromones(test_tsp.dimension, test_tsp.node_coords, pheromones, 20, sf::Color::Red);
		gui.draw_tour(test_tsp.dimension, test_tsp.node_coords, best_tour, 20, sf::Color::Black);
		gui.draw_points(test_tsp.node_coords, 15, sf::Color::Blue);
		window.display();

		cudaMemPrefetchAsync(pheromones, sizeof(float)*test_tsp.dimension*test_tsp.dimension, deviceId);
		run_one_ant<<<number_of_blocks,threads_per_block>>>(number_of_ants, get_time(), test_tsp.dimension, pheromones, pheromones_delta, edge_weights, test_aco.a, test_aco.b, test_aco.p, test_aco.Q, tour_preallocated_memory, tour_length);
		cudaDeviceSynchronize();
		finish_epoch<<<number_of_blocks,threads_per_block>>>(number_of_ants, test_tsp.dimension, pheromones, pheromones_delta, edge_weights, tour_preallocated_memory, test_aco.p);
		cudaDeviceSynchronize();
		cudaMemPrefetchAsync(pheromones, sizeof(float)*test_tsp.dimension*test_tsp.dimension, cudaCpuDeviceId);

		for(int i=0;i<number_of_ants;i++){
			if(tour_length[i] < best_tour_length)
			{ 
				best_tour_length = tour_length[i];
				cudaMemcpy(best_tour, &tour_preallocated_memory[i*test_tsp.dimension], sizeof(int)*test_tsp.dimension, ::cudaMemcpyDeviceToHost);
				cout << "New best tour: " << best_tour_length << ", iteration = " << current_iteration << "\n";
				double current_time = get_time();
				cout << "Time: " << (int)(current_time - initial_time) << " seconds " << (((int)((current_time - initial_time)*1000))%1000 )<< " miliseconds\n";
			}
		}
		current_iteration++;





		// SEQUENTIAL

		//gui.draw_pheromones(test_tsp.dimension, test_tsp.node_coords, test_aco.pheromones, 20, sf::Color::Red);
		//gui.draw_tour(test_tsp.dimension, test_tsp.node_coords, best_tour, 20, sf::Color::Black);
		//gui.draw_points(test_tsp.node_coords, 15, sf::Color::Blue);
		//window.display();

		//for(int i=0;i<number_of_ants;i++){
		//	test_aco.run_one_ant();
		//	//cout << "Iteration " << i << "\n";
		//	//cout << "Tour Length = " << test_aco.get_length_of_tour() << ", ";
		//	if(test_aco.tour_length < best_tour_length)
		//	{ 
		//		best_tour_length = test_aco.tour_length;
		//		memcpy(best_tour, test_aco.tour, sizeof(int)*test_tsp.dimension);
		//		cout << "New best tour: " << best_tour_length << ", iteration = " << current_iteration << "\n";
		//		double current_time = get_time();
		//		cout << "Time: " << (int)(initial_time - current_time) << " seconds " << (((int)(initial_time - current_time)*1000)%1000 )<< " miliseconds\n";
		//	}
		//	//cout << "Best Tour Length = " << best_tour_length << "\n";
		//}
		//test_aco.end_epoch();
		//current_iteration++;

	}
	return 0;
}

