//avoid secure deprecate warnings
#define _CRT_SECURE_NO_DEPRECATE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "NBody.h"
#include "NBodyVisualiser.h"

#define USER_NAME "acp18aaf"		//replace with your username
#define EPSILON 0.000001
#define THREADS_PER_BLOCK 128

//cuda kernal to calculate accelerations for all N bodies
__global__ void calculate_acceleration(float* x, float* y, float* m, float* fx, float* fy, float* ax, float* ay, int n, float g, float softening_powed) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j;
	float d_x, d_y, distance_ij_powed, denominator;
	if (i < n) {
		for (j = 0; j < n; j++) {
			d_x = x[j] - x[i];
			d_y = y[j] - x[i];
			distance_ij_powed = d_x * d_x + d_y * d_y;
			denominator = powf(distance_ij_powed + softening_powed, 1.5);
			fx[i] += m[j] * d_x / denominator;
			fy[i] += m[j] * d_y / denominator;
		}

		ax[i] = fx[i] * g;
		ay[i] = fy[i] * g;
	}
}

//cuda kernal to calculate positions for all N bodies
__global__ void calculate_position(float* ax, float* ay, float* vx, float* vy, float* x, float* y, float t, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		vx[i] = vx[i] + t * ax[i];
		vy[i] = vy[i] + t * ay[i];

		x[i] = x[i] + t * vx[i];
		y[i] = y[i] + t * vy[i];
	}
}

//cuda kernal to calculate activity map for all N bodies
__global__ void calculate_activity(float* x, float* y, float* num, int n, int d) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int row, column;
	if (i < n) {
		row = (int)floor((double)x[i] * d);
		column = (int)floor((double)y[i] * d);

		if (row >= 0 && row < d && column >= 0 && column < d) {
			num[column + row * d] += (float)(1.0 * d / n);
		}
	}
}

void print_help();
void step(void);

//define and initialize host global variables
int N = 0;
int D = 0;
MODE mode;
int I = 0;
char *input_file = NULL;

//arrays to store N-bodies's values
float *x = NULL, *y = NULL, *vx = NULL, *vy = NULL, *m = NULL;

//simulation and activity map
float *fx = NULL, *fy = NULL, *ax = NULL, *ay = NULL, *num = NULL;
int size = 0;

//define device pointers for CUDA mode 
float *d_x = NULL, *d_y = NULL, *d_vx = NULL, *d_vy = NULL, *d_m = NULL;
float *d_fx = NULL, *d_fy = NULL, *d_ax = NULL, *d_ay = NULL, *d_num = NULL;

int main(int argc, char *argv[]) {

	//TODO: Processes the command line arguments
		//argc in the count of the command arguments
		//argv is an array (of length argc) of the arguments. The first argument is always the executable name (including path)

	//processes the command line arguments
	if (argc < 4) {
		printf("Argument error: Not enough arguments.\n\n");
		print_help();
		exit(0);
	}
	
	if(argc >= 4) {
		//get parameter M
		N = atoi(argv[1]);
		if (N <= 0) {
			printf("Argument error: N should be at least 1.\n\n");
			print_help();
			exit(0);
		}

		//get parameter D
		D = atoi(argv[2]);
		if (D <= 0) {
			printf("Argument error: D should be at least 1.\n\n");
			print_help();
			exit(0);
		}

		//get parameter M
		if (!strcmp(argv[3], "CPU")) {
			mode = CPU;
		}
		else if (!strcmp(argv[3], "OPENMP")) {
			mode = OPENMP;
		}
		else if (!strcmp(argv[3], "CUDA")) {
			mode = CUDA;
		}
		else {
			printf("Argument error: M should be CPU or OPENMP.\n\n");
			print_help();
			exit(0);
		}
	}
	
	if (argc >= 5) {
		//check argument -i or -f
		if (strcmp(argv[4], "-i") && strcmp(argv[4], "-f")) {
			printf("Argument error: wrong argument.\n\n");
			print_help();
			exit(0);
		}
	}
	
	if (argc >= 6) {
		//get parameter Iteration
		//check argument -i
		if (!strcmp(argv[4], "-i")) {
			I = atoi(argv[5]);
			if (I <= 0) {
				printf("Argument error: I should be at least 1.\n\n");
				print_help();
				exit(0);
			}
		}
		//check argument -f
		else if (!strcmp(argv[4], "-f")) {
			input_file = argv[5];
		}
	}
	
	if (argc >= 7) {
		//check argument -f
		if (strcmp(argv[6], "-f")) {
			printf("Argument error: wrong argument.\n\n");
			print_help();
			exit(0);
		}
	}

	if (argc >= 8) {
		//get paramter input_file
		input_file = argv[7];
	}

	//TODO: Allocate any heap memory

 	size = sizeof(float) * N;
	x = (float*)malloc(size);
	y = (float*)malloc(size);
	vx = (float*)malloc(size);
	vy = (float*)malloc(size);
	m = (float*)malloc(size);

	fx = (float*)malloc(size);
	fy = (float*)malloc(size);
	ax = (float*)malloc(size);
	ay = (float*)malloc(size);
	num = (float*)malloc(sizeof(float) * D * D);

	//allocate GPU meomory for CUDA mode
	if (mode == CUDA) {
		cudaMalloc((void**) &d_x, size);
		cudaMalloc((void**) &d_y, size);
		cudaMalloc((void**) &d_vx, size);
		cudaMalloc((void**) &d_vy, size);
		cudaMalloc((void**) &d_m, size);

		cudaMalloc((void**)&(d_fx), size);
		cudaMalloc((void**)&(d_fy), size);
		cudaMalloc((void**)&(d_ax), size);
		cudaMalloc((void**)&(d_ay), size);
		cudaMalloc((void**)&(d_num), sizeof(float) * D * D);
	}

	
	//TODO: Depending on program arguments, either read initial data from file or generate random data.

	//read initial data from file
	if (input_file) {
		
		int j = 0;	//count the number of correct bodies in the input file 
		FILE* f = fopen("example_input.nbody", "r");	//open the input file in read mode
		char line[1000] = {0};	//store a line in the input file
		char result[20] = {0};	//store a splitted line 
		

		srand((unsigned)time(NULL));	//help to generate random variable

		if (f == NULL) {
			fprintf(stderr, "Error: Could not open input file.\n");
			exit(-1);
		}
		
		//read the input file line by line
		while (!feof(f))
		{
			memset(line, 0, sizeof(line));
			fgets(line, sizeof(line)-1, f);

			//ignore comment lines starting with '#'
			if (line[0] == '#' || line[0] == '\0') {
				continue;
			}

			if (j <= N-1) {
				//split the line by comma
				int i = 0;	
				char *ch1 = line;
				char *ch2 = result;
				while (*ch1 != '\0') {
					//when comes to the splitter ',' or '\n', get a value 
					if (*ch1 != ',' && *ch1 != '\n') {
						*ch2 = *ch1;
						ch1++;
						ch2++;
					}
					else {
						switch (i++)
						{
						case 0:
							x[j] = (float)atof(result);
							if (x[j] <= EPSILON) x[j] = (float)(rand()*1.0 / RAND_MAX);
							break;
						case 1:
							y[j] = (float)atof(result);
							if (y[j] <= EPSILON) y[j] = (float)(rand()*1.0 / RAND_MAX);
							break;
						case 2:
							vx[j] = (float)atof(result);
							if (vx[j] <= EPSILON) vx[j] = 0.0;
							break;
						case 3:
							vy[j] = (float)atof(result);
							if (vy[j] <= EPSILON) vy[j] = 0.0;
							break;
						case 4:
							m[j] = (float)atof(result);
							if (m[j] <= EPSILON) m[j] = (float)(1.0 / N);
							break;
						}
						//printf("result: %s\n", result);
						memset(result, 0, sizeof(result));
						ch2 = result;
						ch1++;
					}
				} 

				if (i != 5) {
					fprintf(stderr, "Error: comma number should exactly be 4 in line %s\n", line);
					exit(-1);
				}
			}

			j++; //add one to the body count
		}

		//the number of bodies in the input file is less tha N
		if (j != N) {
			fprintf(stderr, "Error: The number of nbodies is wrong, change the value of N.\n");
			exit(-1);
		}

		fclose(f);
	}
	// generate random data
	else {
		for (int i = 0; i < N; i++) {
			x[i] = (float)(rand()*1.0 / RAND_MAX);
			y[i] = (float)(rand()*1.0 / RAND_MAX);
			vx[i] = 0.0;
			vy[i] = 0.0;
			m[i] = (float)(1.0 / N);
		}
	}

	//TODO: Depending on program arguments, either configure and start the visualiser or perform a fixed number of simulation steps (then output the timing results).
	
	// console mode (perform a fixed number of simulation steps)
	if (I) {

		clock_t start_time = clock(); //record the Iteration starting time 
		//implement the Iteration
		int i;
		if (mode == CPU) {
			for (i = 0; i < I; i++) {
				step();
			}
		}
		else if (mode == OPENMP) {
			for (i = 0; i < I; i++) {
				step();
			}
		}
		else if (mode == CUDA) {

			//copy memory from host to device
			cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_vx, vx, size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_vy, vy, size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_m, m, size, cudaMemcpyHostToDevice);

			for (i = 0; i < I; i++) {
				step();
			}
		}
		else {
			fprintf(stderr, "Error: wrong mode.\n");
			exit(-1);
		}
		clock_t end_time = clock(); //record the Iteration ending time

		float run_time = (end_time - start_time) / (float) CLOCKS_PER_SEC;
		int seconds = (int) floor(run_time);
		int milliseconds = (int) ((run_time - seconds)*1000);
		fprintf(stdout, "Execution time %d seconds %d milliseconds\n", seconds, milliseconds);
	}
	// visualisation mode 
	else {
		//input device pointers in CUDA mode for visulization 
		if (mode == CUDA){
			//copy memory from host to device
			cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_vx, vx, size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_vy, vy, size, cudaMemcpyHostToDevice);
			cudaMemcpy(d_m, m, size, cudaMemcpyHostToDevice);

			initViewer(N, D, mode, step);
			setNBodyPositions2f(d_x, d_y);
			setActivityMapData(d_num);
			startVisualisationLoop();
		} else {
			initViewer(N, D, mode, step);
			setNBodyPositions2f(x, y);
			setActivityMapData(num);
			startVisualisationLoop();
		}
	}

	//free allocated host memory
	free(x);
	free(y);
	free(vx);
	free(vy);
	free(m);
	free(fx);
	free(fy);
	free(ax);
	free(ay);
	free(num);

	//free allocated device memory for CUDA mode
	if (mode == CUDA){
		cudaFree(d_x);
		cudaFree(d_y);
		cudaFree(d_vx);
		cudaFree(d_vy);
		cudaFree(d_m);
		cudaFree(d_fx);
		cudaFree(d_fy);
		cudaFree(d_ax);
		cudaFree(d_ay);
		cudaFree(d_num);
	}

	return 0;
}

void step(void)
{
	//TODO: Perform the main simulation of the NBody system

	//initialize fx, fy, num to be 0.0
	memset(fx, 0, size);
	memset(fy, 0, size);
	memset(num, 0, size);

	//calculate this term firstly to avoid to calculate it redundantly
	float softening_powed = (float) (pow(SOFTENING, 2));

	//CPU mode 
	if (mode == CPU) {
		//calculate force and acceleration for all bodies
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				// calculate distance from body j to body i firstly  
				float d_x = x[j] - x[i];
				float d_y = y[j] - y[i];
				float distance_ij_powed = d_x * d_x + d_y * d_y;
				float denominator = (float)pow((double)distance_ij_powed + softening_powed, 3.0 / 2);
				fx[i] += (float)(((double)m[j] * d_x) / denominator);
				fy[i] += (float)(((double)m[j] * d_y) / denominator);
			}

			//calculate acceleration for body i
			ax[i] = fx[i] * G;
			ay[i] = fy[i] * G;
		}

		//calculate motion for all bodies 
		for (int i = 0; i < N; i++) {
			vx[i] = vx[i] + dt * ax[i];
			vy[i] = vy[i] + dt * ay[i];

			//calculate the location for body i
			x[i] = x[i] + dt * vx[i];
			y[i] = y[i] + dt * vy[i];
		}

		//calculate activity map
		//get the location of body i in the one dimensional array with D*D length
		//initialize the count array to 0
		memset(num, 0, sizeof(float) * D * D);

		for (int i = 0; i < N; i++) {
			int row = (int) floor((double)x[i] * D);
			int column = (int) floor((double)y[i] * D);

			if (row >= 0 && row < D && column >= 0 && column < D) {
				num[column + row * D] += (float) (1.0*D / N);
			}
		}
	}
	//OPENMP mode 
	else if (mode == OPENMP) {
		//calculate force and acceleration for all bodies
		int i, j;
		#pragma omp parallel for shared(x,y,vx,vy,m,N,fx,fy,ax,ay)
		for (i = 0; i < N; i++) {
			for (j = 0; j < N; j++) {
				// calculate distance from body j to body i firstly  
				float d_x = x[j] - x[i];
				float d_y = y[j] - y[i];
				float distance_ij_powed = d_x * d_x + d_y * d_y;
				float denominator = (float)pow((double)distance_ij_powed + softening_powed, 3.0 / 2);

				fx[i] += (float)(((double)m[j] * d_x) / denominator);
				fy[i] += (float)(((double)m[j] * d_y) / denominator);
			}

			//calculate acceleration for body i
			ax[i] = G * fx[i];
			ay[i] = G * fy[i];
		}
		#pragma omp barrier

		//calculate motion for all bodies 
		#pragma omp parallel for default(none) shared(vx,vy,x,y,N,ax,ay) private(i)
		for (i = 0; i < N; i++) {
			//calculate the velocity for body i
			vx[i] = vx[i] + dt * ax[i];
			vy[i] = vy[i] + dt * ay[i];

			//calculate the location for body i
			x[i] = x[i] + dt * vx[i];
			y[i] = y[i] + dt * vy[i];
		}
		#pragma omp barrier

		//calculate activity map
		//get the location of body i in the one dimensional array with D*D length
		//initialize the count array to 0
		memset(num, 0, sizeof(float) * D * D);

		#pragma omp parallel for shared(x,y,N,D,num) private(i)
		for (i = 0; i < N; i++) {
			int row = (int)floor((double)x[i] * D);
			int column = (int)floor((double)y[i] * D);

			if (row >= D || row <= 0)	continue;
			if (column >= D || column <= 0)	continue;
			#pragma omp atomic
			num[column + row * D] += (float)(1.0*D / N);
		}
	}
	//CUDA mode
	else if (mode == CUDA) {
		
		int M = THREADS_PER_BLOCK;

		//initialize d_fx, d_fy, d_num to be 0.0
		cudaMemset(d_fx, 0, size);
		cudaMemset(d_fy, 0, size);
		cudaMemset(d_num, 0, size);

		//launch calculate_acceleration kernal in GPU
		calculate_acceleration<<<(N + M -1)/M, M>>>(d_x, d_y, d_m, d_fx, d_fy, d_ax, d_ay, N, G, softening_powed);
		cudaDeviceSynchronize();

		//launch calculate_position kernal in GPU
		calculate_position<<<(N + M -1)/M, M>>>(d_ax, d_ay, d_vx, d_vy, d_x, d_y, dt, N);
		cudaDeviceSynchronize();

		//launch calculate_a kernal in GPU
		calculate_activity<<<(N + M -1)/M, M>>>(d_x, d_y, d_num, N, D);
		cudaDeviceSynchronize();
	}

	else {
		fprintf(stderr, "Error: wrong mode.\n");
		exit(-1);
	}
}



void print_help(){
	printf("nbody_%s N D M [-i I] [-i input_file]\n", USER_NAME);

	printf("where:\n");
	printf("\tN                Is the number of bodies to simulate.\n");
	printf("\tD                Is the integer dimension of the activity grid. The Grid has D*D locations.\n");
	printf("\tM                Is the operation mode, either  'CPU' or 'OPENMP'\n");
	printf("\t[-i I]           Optionally specifies the number of simulation iterations 'I' to perform. Specifying no value will use visualisation mode. \n");
	printf("\t[-f input_file]  Optionally specifies an input file with an initial N bodies of data. If not specified random data will be created.\n");
}
