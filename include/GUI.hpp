#ifndef GUI_HPP
#define GUI_HPP

#include <SFML/Graphics.hpp>
#include <bits/stdc++.h>
#include "./ACO.hpp"
#include "./TSPLIB.hpp"

// Function to calculate the unit vector perpendicular to the line direction
sf::Vector2f perpendicular(const sf::Vector2f& vector);

// Function to draw a thick line
void draw_thick_line(sf::RenderWindow& window, sf::Vector2f start, sf::Vector2f end, float thickness, sf::Color color);

void draw_points(sf::RenderWindow& window, vector<pair<float,float>> points, float size, sf::Color color);

// Set the edges of the viewport
void set_view_edges(sf::RenderWindow& window, float left, float right, float top, float bottom);

// Sets viewport based on extreme points
void set_view_edges_given_points(sf::RenderWindow& window, int dimension, vector<pair<float,float>>);

void draw_pheromones(sf::RenderWindow& window, int dimension, vector<pair<float,float>> node_coords, float * pheromones, float size, sf::Color color);

void draw_tour(sf::RenderWindow& window, int dimension, vector<pair<float,float>> node_coords, int * tour, float size, sf::Color color);

#endif
