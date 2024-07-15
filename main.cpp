#include <SFML/Graphics.hpp>
#include <cmath>
#include <bits/stdc++.h>
#include "./include/ACO.hpp"

#define DEBUG false

using namespace std;


// Function to calculate the unit vector perpendicular to the line direction
sf::Vector2f perpendicular(const sf::Vector2f& vector) {
	return sf::Vector2f(-vector.y, vector.x);
}

// Function to draw a thick line
void drawThickLine(sf::RenderWindow& window, sf::Vector2f start, sf::Vector2f end, float thickness, sf::Color color) {
	sf::Vector2f direction = end - start;
	float length = std::sqrt(direction.x * direction.x + direction.y * direction.y);
	direction /= length; // Normalize direction vector

	sf::Vector2f offset = perpendicular(direction) * (thickness / 2.0f);

	sf::VertexArray quad(sf::Quads, 4);
	quad[0].position = start - offset;
	quad[1].position = start + offset;
	quad[2].position = end + offset;
	quad[3].position = end - offset;

	for (int i = 0; i < 4; ++i) {
		quad[i].color = color;
	}

	window.draw(quad);
}

class VIEW_EDGES{
	public:
		float left;
		float right;
		float top;
		float bottom;
		VIEW_EDGES(float _view_left, float _view_right, float _view_top, float _view_bottom)
		{
			left = _view_left;
			right = _view_right;
			top = _view_top;
			bottom = _view_bottom;
		}
};

pair<float,float> convert_tsplib_point_to_view(sf::RenderWindow& window, pair<int,int> point, VIEW_EDGES view_edges)
{
	float x = (point.first - view_edges.left)/(view_edges.right-view_edges.left) * window.getSize().x;
	float y = (1-(point.second - view_edges.top )/(view_edges.bottom-view_edges.top) ) * window.getSize().y;
	return make_pair(x,y);
}

void draw_points(sf::RenderWindow& window, vector<pair<float,float>> points, VIEW_EDGES view_edges, float size, sf::Color color) {
	sf::CircleShape circle_shape(size);

	circle_shape.setOrigin(circle_shape.getRadius(), circle_shape.getRadius());
	circle_shape.setFillColor(color);
	for(pair<float,float> point : points) {
		float x = (point.first - view_edges.left)/(view_edges.right-view_edges.left) * window.getSize().x;
		float y = (1-(point.second - view_edges.top )/(view_edges.bottom-view_edges.top) ) * window.getSize().y;
		circle_shape.setPosition(x,y);
		window.draw(circle_shape);
	}
}



class TSPLIB_INSTANCE{
	public:
		string name;
		string type;
		string comment;
		int dimension;
		string edge_weight_type;
		vector<pair<int,int>> node_coords;
		TSPLIB_INSTANCE(string filename)
		{
			// Open the file for reading
			std::ifstream file(filename);

			// Check if the file was opened successfully
			if (!file.is_open()) {
				std::cerr << "Failed to open file." << std::endl;
				exit(1);
			}

			// Read from the file line by line
			std::string line;
			bool reached_data_part = false;
			while (std::getline(file, line)) {
				// Do something with the line of text
				if(DEBUG) std::cout << "line: " << line << std::endl;

				std::istringstream iss(line);
				std::string key;
				if (!reached_data_part && std::getline(iss, key, ':')) {
					std::string value;
					if(DEBUG) cout << "KEY IS [" << key << "]\n";
					if(DEBUG) cout << "VALUE IS [" << value << "]\n";
					std::getline(iss >> std::ws, value);
					if (key == "NAME") {
						name = value;
					} else if (key == "TYPE") {
						type = value;
					} else if (key == "COMMENT") {
						comment = value;
					} else if (key == "DIMENSION") {
						dimension = std::stoi(value); // Convert string to integer
					} else if (key == "EDGE_WEIGHT_TYPE ") {
						edge_weight_type = value;
					} else if (key == "NODE_COORD_SECTION") {
						reached_data_part = true;
					}
				}
				else
				{
					if(line == "EOF") break;
					int index, x, y;
					iss >> index >> x >> y;
					if(DEBUG) cout << "POINT " << index << " (" << x << ", " << y << ")\n";
					node_coords.push_back(make_pair(x,y));
				}

			}

			if(reached_data_part == false)
			{
				cout << "ERROR: COULD NOT FIND NODE_COORD_SECTION!" << endl;
				exit(1);
			}

			// Output the parsed values (just for demonstration)
			if(DEBUG) std::cout << "NAME: " << name << std::endl;
			if(DEBUG) std::cout << "TYPE: " << type << std::endl;
			if(DEBUG) std::cout << "COMMENT: " << comment << std::endl;
			if(DEBUG) std::cout << "DIMENSION: " << dimension << std::endl;
			if(DEBUG) std::cout << "EDGE_WEIGHT_TYPE: " << edge_weight_type << std::endl;

			// Close the file
			file.close();

		}

		int euclidean_distance(int node_i_index, int node_j_index)
		{
			pair<int,int> node_i = node_coords[node_i_index];
			pair<int,int> node_j = node_coords[node_j_index];

			// Euclidean distance function defined in the TSPLIB Paper
			int xd = node_i.first - node_j.first;
			int yd = node_i.second - node_j.second;
			int dij = (int) ( sqrt( xd*xd + yd*yd ) + 0.5 );

			if(DEBUG) cout << "Distance from city[" << node_i_index << "] (" << node_i.first << ", " << node_i.second << ") to city[" << node_j_index << "] (" << node_j.first << ", " << node_j.second << ") is equal to " << dij << endl;

			return dij;
		}


};

VIEW_EDGES get_view_edges_of_tsplib_instance(TSPLIB_INSTANCE tsp_instance)
{
	// Get borders of graph
	int smallest_node_x = numeric_limits<int>::max(), greatest_node_x = numeric_limits<int>::min();
	int smallest_node_y = numeric_limits<int>::max(), greatest_node_y = numeric_limits<int>::min();

	for(auto node : tsp_instance.node_coords)
	{
		if( node.first > greatest_node_x) greatest_node_x = node.first;
		if( node.first < smallest_node_x) smallest_node_x = node.first;
		if( node.second > greatest_node_y) greatest_node_y = node.second;
		if( node.second < smallest_node_y) smallest_node_y = node.second;
	}

	float padding_percent = 0.05f;
	float view_left   = smallest_node_x - (greatest_node_x - smallest_node_x) * padding_percent;
	float view_right  = greatest_node_x + (greatest_node_x - smallest_node_x) * padding_percent;
	float view_top    = smallest_node_y - (greatest_node_y - smallest_node_y) * padding_percent;
	float view_bottom = greatest_node_y + (greatest_node_y - smallest_node_y) * padding_percent;

	return VIEW_EDGES(view_left, view_right, view_top, view_bottom);
}

void draw_tsplib_instance(sf::RenderWindow& window, TSPLIB_INSTANCE tsp_instance, float size, sf::Color color) {

	// Convert to float pair vector
	vector<pair<float,float>> points;
	for(pair<int,int> node : tsp_instance.node_coords)
	{
		pair<float, float> point = make_pair(node.first, node.second);
		points.push_back(point);
	}

	VIEW_EDGES view_edges = get_view_edges_of_tsplib_instance(tsp_instance);
	draw_points(window, points, view_edges, size, color);
}


void draw_edges_distances(sf::RenderWindow& window, TSPLIB_INSTANCE tsp_instance, float size, sf::Color color)
{
	VIEW_EDGES view_edges = get_view_edges_of_tsplib_instance(tsp_instance);

	float highest_value = numeric_limits<float>::min();
	float distance;

	for(int i=0;i<tsp_instance.dimension; i++)
	{
		for(int j=0;j<tsp_instance.dimension; j++)
		{
			distance = tsp_instance.euclidean_distance(i,j);
			highest_value = (distance > highest_value)? distance : highest_value;
		}
	}

	for(int i=0;i<tsp_instance.dimension; i++)
	{
		for(int j=0;j<tsp_instance.dimension; j++)
		{
			distance = tsp_instance.euclidean_distance(i,j);

			pair<int,int> node_i = tsp_instance.node_coords[i];
			pair<float, float> node_i_view = convert_tsplib_point_to_view(window, node_i, view_edges);
			pair<int,int> node_j = tsp_instance.node_coords[j];
			pair<float, float> node_j_view = convert_tsplib_point_to_view(window, node_j, view_edges);
			drawThickLine(window, sf::Vector2f(node_i_view.first, node_i_view.second), sf::Vector2f(node_j_view.first, node_j_view.second), distance/highest_value*size, color);
		}
	}
}

void draw_pheromones(sf::RenderWindow& window, TSPLIB_INSTANCE tsp_instance, float * pheromones, float size, sf::Color color)
{
	VIEW_EDGES view_edges = get_view_edges_of_tsplib_instance(tsp_instance);

	float highest_value = numeric_limits<float>::min();
	float distance;

	for(int i=0;i<tsp_instance.dimension; i++)
	{
		for(int j=0;j<tsp_instance.dimension; j++)
		{
			distance = pheromones[i + j*tsp_instance.dimension];
			highest_value = (distance > highest_value)? distance : highest_value;
		}
	}

	for(int i=0;i<tsp_instance.dimension; i++)
	{
		for(int j=0;j<tsp_instance.dimension; j++)
		{
			distance = pheromones[i + j*tsp_instance.dimension];

			pair<int,int> node_i = tsp_instance.node_coords[i];
			pair<float, float> node_i_view = convert_tsplib_point_to_view(window, node_i, view_edges);
			pair<int,int> node_j = tsp_instance.node_coords[j];
			pair<float, float> node_j_view = convert_tsplib_point_to_view(window, node_j, view_edges);
			if(distance/highest_value*size < 0.1) continue;
			drawThickLine(window, sf::Vector2f(node_i_view.first, node_i_view.second), sf::Vector2f(node_j_view.first, node_j_view.second), distance/highest_value*size, color);
		}
	}
}


void draw_tour(sf::RenderWindow& window, TSPLIB_INSTANCE tsp_instance, int * tour, float size, sf::Color color)
{
	VIEW_EDGES view_edges = get_view_edges_of_tsplib_instance(tsp_instance);

	for(int k=0; k<tsp_instance.dimension; k++)
	{
		int i = tour[k];
		int j = tour[(k+1)%tsp_instance.dimension];

		pair<int,int> node_i = tsp_instance.node_coords[i];
		pair<float, float> node_i_view = convert_tsplib_point_to_view(window, node_i, view_edges);
		pair<int,int> node_j = tsp_instance.node_coords[j];
		pair<float, float> node_j_view = convert_tsplib_point_to_view(window, node_j, view_edges);
		drawThickLine(window, sf::Vector2f(node_i_view.first, node_i_view.second), sf::Vector2f(node_j_view.first, node_j_view.second), size, color);
	}
}

int main(int argc, char ** argv)
{
	if(argc < 2) {
		cout << "Please provide the filename of the .tsp file as an argument\n";
		exit(1);
	}

	string tsp_filename(argv[1]);


	sf::RenderWindow window( sf::VideoMode(1980,1080), "TSPLIB Display");

	sf::CircleShape circle_shape(20);

	circle_shape.setPosition(10,10);
	circle_shape.setOrigin(circle_shape.getRadius(), circle_shape.getRadius());
	circle_shape.setFillColor(sf::Color::Red);

	TSPLIB_INSTANCE test_tsp(tsp_filename);


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
		//draw_edges_distances(window, test_tsp, 2, sf::Color::Red);

		for(int i=0;i<number_of_ants;i++){
			test_aco.run_one_ant();
			//cout << "Iteration " << i << "\n";
			//cout << "Tour Length = " << test_aco.get_length_of_tour() << ", ";
			if(test_aco.tour_length < best_tour_length)
			{ 
				best_tour_length = test_aco.tour_length;
				memcpy(best_tour, test_aco.tour, sizeof(int)*test_tsp.dimension);
				cout << "New best tour: " << best_tour_length << "\n";
				for(int j=0;j<test_tsp.dimension;j++)
				{
					cout << best_tour[j] << ((j==test_tsp.dimension-1)? "\n" : ", ");
				}
			}
			//cout << "Best Tour Length = " << best_tour_length << "\n";
		}
		test_aco.end_epoch();

		draw_pheromones(window, test_tsp, test_aco.pheromones, 10, sf::Color::Red);
		draw_tour(window, test_tsp, best_tour, 10, sf::Color::Black);
		draw_tsplib_instance(window, test_tsp, 7*window_area_scale_factor, sf::Color::Blue);
		window.display();
	}
	return 0;
}

