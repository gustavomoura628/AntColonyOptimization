#include <SFML/Graphics.hpp>
#include <cmath>
#include <bits/stdc++.h>

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


class TSPLIB_Instance{
	public:
		string name;
		string type;
		string comment;
		int dimension;
		string edge_weight_type = "test";
		vector<pair<int,int>> node_coords;
		TSPLIB_Instance(string filename)
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
				std::cout << "line: " << line << std::endl;

				std::istringstream iss(line);
				std::string key;
				if (!reached_data_part && std::getline(iss, key, ':')) {
					std::string value;
					cout << "KEY IS [" << key << "]\n";
					cout << "VALUE IS [" << value << "]\n";
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
					cout << "POINT " << index << " (" << x << ", " << y << ")\n";
					node_coords.push_back(make_pair(x,y));
				}

			}

			// Output the parsed values (just for demonstration)
			std::cout << "NAME: " << name << std::endl;
			std::cout << "TYPE: " << type << std::endl;
			std::cout << "COMMENT: " << comment << std::endl;
			std::cout << "DIMENSION: " << dimension << std::endl;
			std::cout << "EDGE_WEIGHT_TYPE: " << edge_weight_type << std::endl;

			// Close the file
			file.close();

		}

		void draw(sf::RenderWindow& window, float size, sf::Color color) {
			// Get borders of graph
			int window_width = window.getSize().x;
			int window_height = window.getSize().y;
			int smallest_node_x = numeric_limits<int>::max(), greatest_node_x = numeric_limits<int>::min();
			int smallest_node_y = numeric_limits<int>::max(), greatest_node_y = numeric_limits<int>::min();

			for(auto node : node_coords)
			{
				if( node.first > greatest_node_x) greatest_node_x = node.first;
				if( node.first < smallest_node_x) smallest_node_x = node.first;
				if( node.second > greatest_node_y) greatest_node_y = node.second;
				if( node.second < smallest_node_y) smallest_node_y = node.second;
			}

			float padding_percent = 0.05f;
			float window_left   = smallest_node_x - (greatest_node_x - smallest_node_x) * padding_percent;
			float window_right  = greatest_node_x + (greatest_node_x - smallest_node_x) * padding_percent;
			float window_top    = smallest_node_y - (greatest_node_y - smallest_node_y) * padding_percent;
			float window_bottom = greatest_node_y + (greatest_node_y - smallest_node_y) * padding_percent;

			//cout << "window left right [ " << window_left << ", " << window_right << " ]\n";
			//cout << "window top bottom [ " << window_top << ", " << window_bottom << " ]\n";
			//cout << "window width = " << window.getSize().x << ", height = " << window.getSize().y << endl;

			// Draw circles
			sf::CircleShape circle_shape(size);
			circle_shape.setFillColor(color);
			for(auto node : node_coords)
			{
				float x = (node.first - window_left)/(window_right-window_left) * window.getSize().x;
				float y = (1-(node.second - window_top )/(window_bottom-window_top) ) * window.getSize().y;
				circle_shape.setPosition(x, y);
				window.draw(circle_shape);
			}
		}

};

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
	circle_shape.setFillColor(sf::Color::Red);

	TSPLIB_Instance test_tsp(tsp_filename);

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
		test_tsp.draw(window, 3*window_area_scale_factor, sf::Color::Blue);
		window.display();
	}
	return 0;
}

