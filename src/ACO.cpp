#include "../include/ACO.hpp"
#include <iostream>

#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <time.h>

using namespace std;

ACO::ACO(int dimension_, float * edge_weights_src, float a_, float b_, float p_, float Q_)
{
    dimension = dimension_;

    edge_weights = (float*)malloc(sizeof(float) * dimension * dimension);
    memcpy(edge_weights, edge_weights_src, sizeof(float) * dimension * dimension);

    pheromones = (float*)malloc(sizeof(float) * dimension * dimension); 
    // Initialize pheromones
    for(int i = 0; i<dimension*dimension; i++)
    {
        pheromones[i] = 1;
    }

    pheromones_aux = (float*)malloc(sizeof(float) * dimension * dimension);
    memset(pheromones_aux, 0, sizeof(float)*dimension*dimension);

    remaining_cities = (int*)malloc(sizeof(int) * dimension);

    tour = (int*)malloc(sizeof(int) * dimension);

    a = a_;
    b = b_;
    p = p_;
    Q = Q_;

    srand(time(0));
}

float ACO::desire_of_moving_from_city_i_to_j(int i, int j)
{
    float pheromone_i_to_j = pheromones[i + j*dimension];
    float edge_weight_i_to_j = edge_weights[i + j*dimension];

    return pow( pheromone_i_to_j, a ) * pow( 1.0/edge_weight_i_to_j, b );
}

float ACO::probability_ant_moves_from_city_i_to_j(int i, int j)
{
    float total_desire = 0;
    for(int k=0; k<number_of_remaining_cities; k++)
    {
        total_desire += desire_of_moving_from_city_i_to_j(i,remaining_cities[k]);
    }

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
    return length;
}

void ACO::update_pheromones()
{
    float length = get_length_of_tour();
    float delta_pheromone = Q/length;
    for(int k=0; k<dimension; k++)
    {
        int i = tour[k];
        int j = tour[(k+1)%dimension];
        pheromones_aux[i+j*dimension] += delta_pheromone;
    }
}

void ACO::end_epoch()
{
    for(int i=0;i<dimension*dimension;i++)
    {
        pheromones[i] = pheromones[i]*(1-p) + pheromones_aux[i];
    }
    memset(pheromones_aux,0,sizeof(float)*dimension*dimension);
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
    int random_index = rand() % dimension;
    int current_city = remaining_cities[random_index];
    int current_city_index = random_index;
    tour[0] = current_city;
    swap(remaining_cities[current_city_index], remaining_cities[number_of_remaining_cities-1]);
    number_of_remaining_cities--;

    //cout << "Starting city = " << current_city << "\n";

    //for(int i=0; i<number_of_remaining_cities; i++)
    //{
    //    cout << "Desire to go from " << current_city << " to " << remaining_cities[i] << " = " << desire_of_moving_from_city_i_to_j(current_city, remaining_cities[i]) << "\n";
    //    cout << "Probability to go from " << current_city << " to " << remaining_cities[i] << " = " << probability_ant_moves_from_city_i_to_j(current_city, remaining_cities[i]) << "\n";
    //}

    while(number_of_remaining_cities > 0)
    {
        float random_number = (float)rand() / numeric_limits<int>::max();
        float probability_sum = 0;
        int i = 0;
        int destination_city;
        int destination_city_index;
        // ACO Algorithm
        destination_city = remaining_cities[number_of_remaining_cities-1];
        destination_city_index = number_of_remaining_cities-1;
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
        

        // GREEDY TEST
        //float max_probability = 0;
        //for(int i=0;i<number_of_remaining_cities; i++)
        //{
        //    float probability = probability_ant_moves_from_city_i_to_j(current_city, remaining_cities[i]);
        //    if(probability > max_probability)
        //    {
        //        max_probability = probability;
        //        destination_city = remaining_cities[i];
        //        destination_city_index = i;
        //    }
        //}

        //cout << "Destination city = " << destination_city << "\n";
        tour[dimension - number_of_remaining_cities] = destination_city;
        swap(remaining_cities[destination_city_index], remaining_cities[number_of_remaining_cities-1]);
        current_city = destination_city;
        current_city_index = number_of_remaining_cities-1;
        number_of_remaining_cities--;
        //cout << "remaining_cities: {";
        //for(int i=0;i<number_of_remaining_cities;i++)
        //{
        //    cout << "[" << i << "]" << remaining_cities[i] << ((i==number_of_remaining_cities-1)? "}\n" : ", ");
        //}
    }
    //cout << "Tour = {";
    //for(int i=0; i<dimension; i++)
    //{
    //    cout << tour[i] << ((i==dimension-1)? "}\n" : ", ");
    //}
    //cout << "Length of tour = " << get_length_of_tour() << "\n";
    update_pheromones();
}