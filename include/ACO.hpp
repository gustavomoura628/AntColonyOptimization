#ifndef ACO_HPP
#define ACO_HPP

#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdlib>

class ACO{
    public:
        int dimension;
        float * pheromones;
        float * pheromones_aux;
        float * edge_weights;
        float a; // Pheromone Coefficient
        float b; // Edge Weight Coefficient
        float p; // Evaporation Coefficient
        float Q; // Pheromone over Length Constant
        int * remaining_cities;
        int number_of_remaining_cities;
        int * tour;

        ACO(int dimension, float * edge_weights_src, float a, float b, float p, float Q);
        float desire_of_moving_from_city_i_to_j(int i, int j);
        float probability_ant_moves_from_city_i_to_j(int i, int j);
        float get_length_of_tour();
        void update_pheromones();
        void end_epoch();
        void run_one_ant();
};

#endif