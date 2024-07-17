#include <SFML/Graphics.hpp>
#include <cmath>
#include <bits/stdc++.h>
#include "./include/ACO.hpp"
#include "./include/GUI.hpp"
#include "./include/TSPLIB.hpp"

using namespace std;

int main(int argc, char ** argv)
{
	if(argc < 2) {
		cout << "Please provide the filename of the .tsp file as an argument\n";
		exit(1);
	}

	struct timespec timer_init;
	timespec_get(&timer_init, TIME_UTC);

	string tsp_filename(argv[1]);


	sf::RenderWindow window( sf::VideoMode(1980,1080), "TSPLIB Display");

	sf::CircleShape circle_shape(20);

	circle_shape.setPosition(50,100);
	circle_shape.setOrigin(circle_shape.getRadius(), circle_shape.getRadius());
	circle_shape.setFillColor(sf::Color::Red);

	TSPLIB_INSTANCE test_tsp(tsp_filename);

	//set_view_edges_with_tsplib_instance(window, test_tsp);
	set_view_edges_given_points(window, test_tsp.dimension, test_tsp.node_coords);


	float * tsp_edge_weights = (float*)malloc(sizeof(float)*test_tsp.dimension * test_tsp.dimension);
	for(int i=0;i<test_tsp.dimension; i++)
	{
		for(int j=0;j<test_tsp.dimension; j++)
		{
			tsp_edge_weights[i+j*test_tsp.dimension] = test_tsp.euclidean_distance(i,j);
		}

	}

	int number_of_ants = 1000;
	ACO test_aco(test_tsp.dimension, tsp_edge_weights, 1, 5, 0.5, 1000.0/number_of_ants);

	float best_tour_length = numeric_limits<float>::max();
	int * best_tour = (int*)malloc(sizeof(int)*test_tsp.dimension);
	for(int i=0;i<test_tsp.dimension;i++)
	{
		best_tour[i]=i;
	}

	int current_iteration = 0;
	
	while (window.isOpen()) 
	{
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type ==
			sf::Event::Closed)
				window.close();

			//// catch the resize events
			//if (event.type == sf::Event::Resized)
			//{
			//	// update the view to the new size of the window
			//	sf::FloatRect visibleArea(0, 0, event.size.width, event.size.height);
			//	window.setView(sf::View(visibleArea));
			//}
		}

		window.clear(sf::Color::White);
		//window.draw(circle_shape);
		//drawThickLine(window, sf::Vector2f(100, 150), sf::Vector2f(250, 250), 10.0f, sf::Color::Red);
		float window_area_scale_factor = min(window.getSize().x, window.getSize().y) / 1000.0;
		//test_tsp.draw(window, 7*window_area_scale_factor, sf::Color::Blue);

		for(int i=0;i<number_of_ants;i++){
			test_aco.run_one_ant();
			//cout << "Iteration " << i << "\n";
			//cout << "Tour Length = " << test_aco.get_length_of_tour() << ", ";
			if(test_aco.tour_length < best_tour_length)
			{ 
				best_tour_length = test_aco.tour_length;
				memcpy(best_tour, test_aco.tour, sizeof(int)*test_tsp.dimension);
				cout << "New best tour: " << best_tour_length << ", iteration = " << current_iteration << "\n";
				struct timespec current_time;
				timespec_get(&current_time, TIME_UTC);
				double timer_init_float = timer_init.tv_sec * 1000 + (double)timer_init.tv_nsec / 1000000;
				double current_time_float = current_time.tv_sec * 1000 + (double)current_time.tv_nsec / 1000000;
				cout << "Time: " << (int)((current_time_float - timer_init_float)/1000) << " seconds " << ((int)(current_time_float - timer_init_float)%1000 )<< " miliseconds\n";
				//for(int j=0;j<test_tsp.dimension;j++)
				//{
				//	cout << best_tour[j] << ((j==test_tsp.dimension-1)? "\n" : ", ");
				//}
			}
			//cout << "Best Tour Length = " << best_tour_length << "\n";
		}
		test_aco.end_epoch();
		current_iteration++;

		draw_pheromones(window, test_tsp.dimension, test_tsp.node_coords, test_aco.pheromones, 20, sf::Color::Red);
		draw_tour(window, test_tsp.dimension, test_tsp.node_coords, best_tour, 20, sf::Color::Black);
		draw_points(window, test_tsp.node_coords, 15, sf::Color::Blue);
		window.display();
	}
	return 0;
}

