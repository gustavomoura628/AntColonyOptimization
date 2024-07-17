#ifndef TSPLIB_HPP
#define TSPLIB_HPP

#include <bits/stdc++.h>

using namespace std;

class TSPLIB_INSTANCE{
	public:
		string name;
		string type;
		string comment;
		int dimension;
		string edge_weight_type;
		vector<pair<float,float>> node_coords;
		TSPLIB_INSTANCE(string filename);
		int euclidean_distance(int node_i_index, int node_j_index);
};

#endif
