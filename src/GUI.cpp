#include "../include/GUI.hpp"

#include <SFML/Graphics.hpp>
#include <bits/stdc++.h>
#include "../include/ACO.hpp"
#include "../include/TSPLIB.hpp"

using namespace std;

// Function to calculate the unit vector perpendicular to the line direction
sf::Vector2f perpendicular(const sf::Vector2f& vector) {
	return sf::Vector2f(-vector.y, vector.x);
}

// Function to draw a thick line
void draw_thick_line(sf::RenderWindow& window, sf::Vector2f start, sf::Vector2f end, float thickness, sf::Color color) {
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


void draw_points(sf::RenderWindow& window, vector<pair<float,float>> points, float size, sf::Color color) {
	sf::CircleShape circle_shape(size);

	circle_shape.setOrigin(circle_shape.getRadius(), circle_shape.getRadius());
	circle_shape.setFillColor(color);
	for(pair<float,float> point : points) {
		float x = point.first;
		float y = point.second;
		circle_shape.setPosition(x,y);
		window.draw(circle_shape);
	}
}

// Set the edges of the viewport
void set_view_edges(sf::RenderWindow& window, float left, float right, float top, float bottom)
{
    sf::FloatRect visibleArea(left, top, right-left, bottom-top);
    window.setView(sf::View(visibleArea));
}

// Sets viewport based on extreme points
void set_view_edges_given_points(sf::RenderWindow& window, int dimension, vector<pair<float,float>> points)
{
	// Get borders of graph
	float smallest_node_x = numeric_limits<float>::max(), greatest_node_x = numeric_limits<float>::min();
	float smallest_node_y = numeric_limits<float>::max(), greatest_node_y = numeric_limits<float>::min();

	for(pair<float,float> node : points)
	{
		if( node.first > greatest_node_x) greatest_node_x = node.first;
		if( node.first < smallest_node_x) smallest_node_x = node.first;
		if( node.second > greatest_node_y) greatest_node_y = node.second;
		if( node.second < smallest_node_y) smallest_node_y = node.second;
	}

	float padding_percent = 0.1f;
	float view_left   = smallest_node_x - (greatest_node_x - smallest_node_x) * padding_percent/2;
	float view_right  = greatest_node_x + (greatest_node_x - smallest_node_x) * padding_percent/2;
	float view_top    = smallest_node_y - (greatest_node_y - smallest_node_y) * padding_percent/2;
	float view_bottom = greatest_node_y + (greatest_node_y - smallest_node_y) * padding_percent/2;

    set_view_edges(window, view_left, view_right, view_bottom, view_top); // top/bottom flipped, flips y axis of view
}

void draw_pheromones(sf::RenderWindow& window, int dimension, vector<pair<float,float>> node_coords, float * pheromones, float size, sf::Color color)
{
	float distance;
	float highest_value = *max_element(pheromones, pheromones + dimension*dimension);
    float line_width;

	for(int i=0;i<dimension; i++)
	{
		for(int j=0;j<dimension; j++)
		{
			distance = pheromones[i + j*dimension];
            line_width = distance/highest_value*size;
			if(line_width < 1) continue; // Ignores really thin lines

            sf::Vector2f node_i(node_coords[i].first, node_coords[i].second);
            sf::Vector2f node_j(node_coords[j].first, node_coords[j].second);

			draw_thick_line(window, node_i, node_j, line_width, color);
		}
	}
}


void draw_tour(sf::RenderWindow& window, int dimension, vector<pair<float,float>> node_coords, int * tour, float size, sf::Color color)
{
	for(int k=0; k<dimension; k++)
	{
		int i = tour[k];
		int j = tour[(k+1)%dimension];

		sf::Vector2f node_i(node_coords[i].first, node_coords[i].second);
		sf::Vector2f node_j(node_coords[j].first, node_coords[j].second);
		draw_thick_line(window, node_i, node_j, size, color);
	}
}
