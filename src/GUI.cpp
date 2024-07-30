#include "../include/GUI.hpp"

#include <SFML/Graphics.hpp>
#include <bits/stdc++.h>
#include "../include/ACO.hpp"
#include "../include/TSPLIB.hpp"

using namespace std;

GUI::GUI(sf::RenderWindow& window)
: window(window)
{
	camera_view = window.getView(); // default view
}

// Set the edges of the viewport
void GUI::set_camera_view_edges(float left, float right, float top, float bottom)
{
    sf::FloatRect visibleArea(left, top, right-left, bottom-top);
    camera_view = sf::View(visibleArea);
}


// Sets viewport based on extreme points
void GUI::set_camera_view_edges_given_points(int dimension, vector<pair<float,float>> points)
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

    set_camera_view_edges(view_left, view_right, view_bottom, view_top); // top/bottom flipped, flips y axis of view
}

sf::Vector2f GUI::convert_float_pair_to_sf_vector(pair<float,float> point)
{
	return sf::Vector2f(point.first, point.second);
}

sf::Vector2f GUI::convert_point_to_another_view(sf::View dest, sf::View src, sf::Vector2f& point)
{
	sf::Vector2f transformed_point;
	transformed_point.x = (point.x - src.getCenter().x)*(dest.getSize().x/src.getSize().x) + dest.getCenter().x;
	transformed_point.y = (point.y - src.getCenter().y)*(dest.getSize().y/src.getSize().y) + dest.getCenter().y;

	return transformed_point;
}

float GUI::get_window_size_scaling_factor()
{
	return min(window.getSize().x, window.getSize().y)*0.0005;
}

// Function to calculate the unit vector perpendicular to the line direction
sf::Vector2f GUI::perpendicular(const sf::Vector2f& vector) {
	return sf::Vector2f(-vector.y, vector.x);
}

// Function to draw a thick line
void GUI::draw_thick_line(sf::Vector2f start, sf::Vector2f end, float thickness, sf::Color color) {
	thickness*=get_window_size_scaling_factor();
	start = convert_point_to_another_view(window.getView(), camera_view, start);
	end = convert_point_to_another_view(window.getView(), camera_view, end);

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

void GUI::draw_edge(sf::Vector2f start, sf::Vector2f end, float thickness, sf::Color color) 
{
	// EUCLIDEAN:
	draw_thick_line(start, end, thickness, color);

	return;
	// L1 hORM: 
	float max_height;
	sf::Vector2f top, bottom;
	if(start.y > end.y)
	{
		top = start;
		bottom = end;
	}else{
		top = end;
		bottom = start;
	}
	sf::Vector2f middle;
	middle.x = bottom.x;
	middle.y = top.y;
	draw_thick_line(top, middle, thickness, color);
	draw_thick_line(middle, bottom, thickness, color);

	// circle in the middle to make curve smooth
	draw_point(middle, thickness/2, color);
}

void GUI::draw_point(sf::Vector2f point_sf, float size, sf::Color color) {
	size*=get_window_size_scaling_factor();
	sf::CircleShape circle_shape(size);

	circle_shape.setOrigin(circle_shape.getRadius(), circle_shape.getRadius());
	circle_shape.setFillColor(color);
	sf::Vector2f point_transformed = convert_point_to_another_view(window.getView(), camera_view, point_sf);
	circle_shape.setPosition(point_transformed);
	window.draw(circle_shape);
}

void GUI::draw_points(vector<pair<float,float>> points, float size, sf::Color color) {
	size*=get_window_size_scaling_factor();
	sf::CircleShape circle_shape(size);

	circle_shape.setOrigin(circle_shape.getRadius(), circle_shape.getRadius());
	circle_shape.setFillColor(color);
	for(pair<float,float> point : points) {
		sf::Vector2f point_sf = convert_float_pair_to_sf_vector(point);
		sf::Vector2f point_transformed = convert_point_to_another_view(window.getView(), camera_view, point_sf);

		circle_shape.setPosition(point_transformed);

		window.draw(circle_shape);
	}
}

void GUI::draw_pheromones(int dimension, vector<pair<float,float>> node_coords, float * pheromones, float size, sf::Color color)
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

			draw_edge(node_i, node_j, line_width, color);
		}
	}
}

void GUI::draw_tour(int dimension, vector<pair<float,float>> node_coords, int * tour, float size, sf::Color color)
{
	for(int k=0; k<dimension; k++)
	{
		int i = tour[k];
		int j = tour[(k+1)%dimension];

		sf::Vector2f node_i(node_coords[i].first, node_coords[i].second);
		sf::Vector2f node_j(node_coords[j].first, node_coords[j].second);
		draw_edge(node_i, node_j, size, color);
	}
}
