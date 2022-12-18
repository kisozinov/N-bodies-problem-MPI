#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <random>
#include <fstream>
#include <string>
#include <filesystem>
#include <iostream>
#include "mpi.h"

typedef struct body
{
	double px, py;
	double vx, vy;
	double ax, ay;
	long m;
}body;

body bodies[256];
const int N = 256;
const double G = 6.67E-11;// G - гравитационная постоянная
int timestep = 100;
double delta_t = 0.0;
double r = 0.01; // радиус тела
int n_iterations = 20000;
FILE* fp;
const char* filename[4] = { "result1.csv","result2.csv","result3.csv","result4.csv" };

void calc_force(int index)
{
	bodies[index].ax = 0;
	bodies[index].ay = 0;
	for (int i = 0; i < N; i++)
	{
		if (i != index)
		{
			double dx = bodies[i].px - bodies[index].px;
			double dy = bodies[i].py - bodies[index].py;
			double d = (dx * dx + dy * dy);
			if (d < r * r)d = r * r;
			d *= sqrt(d);//^(3/2)

			bodies[index].ax += G * bodies[index].m * (dx) / d;
			bodies[index].ay += G * bodies[index].m * (dy) / d;
			//printf("%lf %lf  ",dx,dy);
		}

	}
}
void calc_velocities(int index)
{

	bodies[index].vx += bodies[index].ax * delta_t;
	bodies[index].vy += bodies[index].ay * delta_t;
}
void calc_positions(int index)
{

	bodies[index].px += bodies[index].vx * delta_t;
	bodies[index].py += bodies[index].vy * delta_t;
}

void print(std::ofstream &fp, int i)
{
	fp << "t" << i << ',';
	for (int i = 0; i < N; i++)
	{
		//fprintf(fp, "%lf,", bodies[i].px);
		fp << std::to_string(bodies[i].px) << ",";
		
	}
	for (int i = 0; i < N; i++)
	{
		fp << std::to_string(bodies[i].py) << ",";
	}
	fp << "\n";
}

int makefiles() {

	// Массы
	std::ofstream out;
	out.open("E:\\Projects_C_C++\\N_bodies\\N_bodies\\data\\weights.txt");
	if (out.is_open()) {
		for (size_t i = 0; i < N; ++i) {
			out << rand() % 20000 + 500 << std::endl;
		}
	}
	out.close();
	out.clear();

	// Координаты
	out.open("E:\\Projects_C_C++\\N_bodies\\N_bodies\\data\\coordinates.txt");
	if (out.is_open()) {
		for (size_t i = 0; i < N; ++i) {
			out << static_cast <float> (rand()) / static_cast <float> (RAND_MAX) << std::endl;
			out << static_cast <float> (rand()) / static_cast <float> (RAND_MAX) << std::endl;
		}
	}
	out.close();
	out.clear();

	// Скорости
	out.open("E:\\Projects_C_C++\\N_bodies\\N_bodies\\data\\velocities.txt");
	if (out.is_open()) {
		for (size_t i = 0; i < N; ++i) {
			out << 0 << std::endl;
		}
	}
	out.close();
	return 0;
}

int main(int argc, char* argv[])
{
	//init
	int a = makefiles();
	delta_t = 1.0 / timestep;

	// Считывание данных из файла
	std::string line;
	std::ifstream in_m("E:\\Projects_C_C++\\N_bodies\\N_bodies\\data\\weights.txt");
	std::ifstream in_v("E:\\Projects_C_C++\\N_bodies\\N_bodies\\data\\velocities.txt");
	std::ifstream in_coord("E:\\Projects_C_C++\\N_bodies\\N_bodies\\data\\coordinates.txt");

	for (int i = 0; i < N; i++)
	{
		if (in_m.is_open() && getline(in_m, line)) {
			bodies[i].m = std::stoi(line);
		}
	}
	in_m.clear();
	in_m.close();
	for (int i = 0; i < N; i++) {
		if (in_v.is_open() && getline(in_v, line)) {
			if (line == (std::string)"0") {
				bodies[i].vx = (int)0;
				bodies[i].vy = (int)0;
			}
			else {
				bodies[i].vx = std::stod(line);
				bodies[i].vy = std::stod(line);
			}
			
		}
		bodies[i].ax = 0;
		bodies[i].ay = 0;
	}
	in_v.clear();
	in_v.close();
	for (int i = 0; i < N; i++) {
		if (in_coord.is_open()) {
			getline(in_coord, line);
			bodies[i].px = std::stod(line);
			getline(in_coord, line);
			bodies[i].py = std::stod(line);
		}
	}
	in_coord.close();
	
	int myid, numprocs;
	clock_t starttime, endtime, startprint, endprint;
	double totalprint, totalwork;


	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	double* mpi_buffer = (double*)malloc(sizeof(double) * 1000000);
	MPI_Buffer_attach(mpi_buffer, sizeof(double) * 1000000);

	std::ofstream fp;
	fp.open(filename[numprocs - 1]);

	starttime = clock();
	totalprint = 0;

	for (int i = 0; i < n_iterations; i++)
	{
		for (int j = 0; j < numprocs; j++)
		{
			if (j != myid)
				MPI_Bsend((bodies + (N / numprocs) * myid), sizeof(body) * N / numprocs, MPI_BYTE, j, i * 10 + myid, MPI_COMM_WORLD);
		}
		for (int j = 0; j < numprocs; j++)
		{
			if (j != myid)
			{
				MPI_Status status;
				MPI_Recv((bodies + (N / numprocs) * j), sizeof(body) * N / numprocs, MPI_BYTE, j, i * 10 + j, MPI_COMM_WORLD, &status);
			}
		}
		for (int j = (N / numprocs) * myid; j < (N / numprocs) * (myid + 1); j++)
		{
			calc_force(j);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		for (int j = (N / numprocs) * myid; j < (N / numprocs) * (myid + 1); j++)
		{
			calc_velocities(j);
			calc_positions(j);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		startprint = clock();
		print(fp, i);
		endprint = clock();
		totalprint += (double)(endprint - startprint);
	}

	endtime = clock();
	totalwork = (double)(endtime - starttime) - totalprint;
	printf("rank=%d time:%lf\n", myid, totalwork / CLOCKS_PER_SEC);

	if (myid != 0)
	{
		MPI_Send((bodies + (N / numprocs) * myid), sizeof(body) * N / numprocs, MPI_BYTE, 0, myid, MPI_COMM_WORLD);
	}
	if (myid == 0)
	{
		for (int i = 1; i < numprocs; i++)
		{
			MPI_Status status;
			MPI_Recv((bodies + (N / numprocs) * i), sizeof(body) * N / numprocs, MPI_BYTE, i, i, MPI_COMM_WORLD, &status);
			
		}
	}

	MPI_Finalize();
	return 0;
}
