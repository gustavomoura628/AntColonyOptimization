#include "../include/ACO.hpp"

#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <time.h>

using namespace std;

ACO::ACO(int dimension_, float * edge_weights_src, float a_, float b_, float p_, float Q_)
{
    a = a_;
    b = b_;
    p = p_;
    Q = Q_;

    dimension = dimension_;

    edge_weights = (float*)malloc(sizeof(float) * dimension * dimension);
    memcpy(edge_weights, edge_weights_src, sizeof(float) * dimension * dimension);

    pheromones = (float*)malloc(sizeof(float) * dimension * dimension); 
    // Initialize pheromones
    for(int i = 0; i<dimension*dimension; i++)
    {
        pheromones[i] = 1;
    }

    pheromones_delta = (float*)malloc(sizeof(float) * dimension * dimension);
    memset(pheromones_delta, 0, sizeof(float)*dimension*dimension);

    remaining_cities = (int*)malloc(sizeof(int) * dimension);

    tour = (int*)malloc(sizeof(int) * dimension);

    srand(time(0));
}

float ACO::desire_of_moving_from_city_i_to_j(int i, int j)
{
    float pheromone_i_to_j = pheromones[i + j*dimension];
    float edge_weight_i_to_j = edge_weights[i + j*dimension];

    return pow( pheromone_i_to_j, a ) * pow( 1.0/edge_weight_i_to_j, b );
}

float ACO::calculate_total_desire_from_i(int i)
{
    total_desire = 0;
    for(int k=0; k<number_of_remaining_cities; k++)
    {
        total_desire += desire_of_moving_from_city_i_to_j(i,remaining_cities[k]);
    }
    return total_desire;
}

float ACO::probability_ant_moves_from_city_i_to_j(int i, int j)
{
    return desire_of_moving_from_city_i_to_j(i,j)/total_desire;
}

float ACO::get_length_of_tour()
{
    float length = 0;
    for(int k=0; k<dimension; k++)
    {
        int i = tour[k];
        int j = tour[(k+1)%dimension];
        length += edge_weights[i + j*dimension];
    }
    tour_length = length;
    return length;
}

void ACO::update_pheromones()
{
    float length = tour_length;
    float delta_pheromone = Q/length;
    for(int k=0; k<dimension; k++)
    {
        int i = tour[k];
        int j = tour[(k+1)%dimension];
        pheromones_delta[i+j*dimension] += delta_pheromone;
    }
}

void ACO::end_epoch()
{
    for(int i=0;i<dimension*dimension;i++)
    {
        pheromones[i] = pheromones[i]*(1-p) + pheromones_delta[i];
    }
    memset(pheromones_delta,0,sizeof(float)*dimension*dimension);
}

void ACO::run_one_ant()
{
    // Set up
    number_of_remaining_cities = dimension;
    for(int i=0;i<dimension;i++)
    {
        remaining_cities[i] = i;
    }

    // Create tour
    // Initialize first city
    int current_city_index = rand() % dimension;
    int current_city = remaining_cities[current_city_index];

    while(number_of_remaining_cities > 0)
    {
        tour[dimension - number_of_remaining_cities] = current_city;
        swap(remaining_cities[current_city_index], remaining_cities[number_of_remaining_cities-1]);
        number_of_remaining_cities--;

        calculate_total_desire_from_i(current_city);
        float random_number = (float)rand() / numeric_limits<int>::max();
        float probability_sum = 0;
        int i = 0;
        // ACO Algorithm
        int destination_city = remaining_cities[number_of_remaining_cities-1];
        int destination_city_index = number_of_remaining_cities-1;
        for(;i<number_of_remaining_cities; i++)
        {
            probability_sum += probability_ant_moves_from_city_i_to_j(current_city, remaining_cities[i]);
            if(probability_sum > random_number)
            {
                destination_city = remaining_cities[i];
                destination_city_index = i;
                break;
            }
        }
        current_city = destination_city;
        current_city_index = destination_city_index;
    }
    get_length_of_tour();
    update_pheromones();
}
