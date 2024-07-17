#include "../include/TSPLIB.hpp"
#include <bits/stdc++.h>

using namespace std;

TSPLIB_INSTANCE::TSPLIB_INSTANCE(string filename)
{
    // Open the file for reading
    ifstream file(filename);

    // Check if the file was opened successfully
    if (!file.is_open()) {
        cerr << "Failed to open file." << endl;
        exit(1);
    }

    // Read from the file line by line
    string line;
    bool reached_data_part = false;
    while (getline(file, line)) {
        // Do something with the line of text
        //cout << "line: " << line << endl;

        istringstream iss(line);
        string key;
        if (!reached_data_part && getline(iss, key, ':')) {
            while(key.back() == ' ') key.pop_back(); // remove trailing whitespace
            string value;
            getline(iss >> ws, value);
            if (key == "NAME") {
                name = value;
            } else if (key == "TYPE") {
                type = value;
            } else if (key == "COMMENT") {
                comment = value;
            } else if (key == "DIMENSION") {
                dimension = stoi(value); // Convert string to integer
            } else if (key == "EDGE_WEIGHT_TYPE") {
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
            node_coords.push_back(make_pair(x,y));
        }

    }

    if(reached_data_part == false)
    {
        cerr << "ERROR: COULD NOT FIND NODE_COORD_SECTION!" << endl;
        exit(1);
    }

    // Output the parsed values
    //cout << "NAME: " << name << endl;
    //cout << "TYPE: " << type << endl;
    //cout << "COMMENT: " << comment << endl;
    //cout << "DIMENSION: " << dimension << endl;
    //cout << "EDGE_WEIGHT_TYPE: " << edge_weight_type << endl;

    // Close the file
    file.close();

}

int TSPLIB_INSTANCE::euclidean_distance(int node_i_index, int node_j_index)
{
    pair<float,float> node_i = node_coords[node_i_index];
    pair<float,float> node_j = node_coords[node_j_index];

    // Euclidean distance function defined in the TSPLIB Paper
    int xd = node_i.first - node_j.first;
    int yd = node_i.second - node_j.second;
    int dij = (int) ( sqrt( xd*xd + yd*yd ) + 0.5 );

    ////cout << "Distance from city[" << node_i_index << "] (" << node_i.first << ", " << node_i.second << ") to city[" << node_j_index << "] (" << node_j.first << ", " << node_j.second << ") is equal to " << dij << endl;

    return dij;
}
