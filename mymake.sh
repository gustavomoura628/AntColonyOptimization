#g++ main.cpp src/ACO.cpp src/GUI.cpp src/TSPLIB.cpp -o main -lsfml-graphics -lsfml-window -lsfml-system
nvcc main.cu src/ACO.cpp src/GUI.cpp src/TSPLIB.cpp -o main -lsfml-graphics -lsfml-window -lsfml-system
