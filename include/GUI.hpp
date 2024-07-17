#ifndef GUI_HPP
#define GUI_HPP

#include <SFML/Graphics.hpp>
#include <bits/stdc++.h>
#include "./ACO.hpp"
#include "./TSPLIB.hpp"


class GUI{
    public:
        sf::RenderWindow& window;
        sf::View camera_view;

        GUI(sf::RenderWindow& window);

        // Set the edges of the viewport
        void set_camera_view_edges(float left, float right, float top, float bottom);

        // Sets viewport based on extreme points
        void set_camera_view_edges_given_points(int dimension, vector<pair<float,float>>);

        sf::Vector2f convert_float_pair_to_sf_vector(pair<float,float> point);

        sf::Vector2f convert_point_to_another_view(sf::View dest, sf::View src, sf::Vector2f& point);

        float get_window_size_scaling_factor();

        // Function to calculate the unit vector perpendicular to the line direction
        sf::Vector2f perpendicular(const sf::Vector2f& vector);

        // Function to draw a thick line
        void draw_thick_line(sf::Vector2f start, sf::Vector2f end, float thickness, sf::Color color);

        void draw_points(vector<pair<float,float>> points, float size, sf::Color color);

        void draw_pheromones(int dimension, vector<pair<float,float>> node_coords, float * pheromones, float size, sf::Color color);

        void draw_tour(int dimension, vector<pair<float,float>> node_coords, int * tour, float size, sf::Color color);

};

#endif
