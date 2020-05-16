//avoid secure deprecate warnings
#define _CRT_SECURE_NO_DEPRECATE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

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

//define and initialize global variables
int N = 0;
int D = 0;
MODE mode = CPU;
int I = 0;
char *input_file = NULL;
nbody_soa *nbodies = NULL;
float *fx = NULL, *fy = NULL, *ax = NULL, *ay = NULL, *num = NULL;

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

	nbodies = (struct nbody_soa*)malloc(sizeof(struct nbody_soa));
	nbodies->x = (float*)malloc(sizeof(float) * N);
	nbodies->y = (float*)malloc(sizeof(float) * N);
	nbodies->vx = (float*)malloc(sizeof(float) * N);
	nbodies->vy = (float*)malloc(sizeof(float) * N);
	nbodies->m = (float*)malloc(sizeof(float) * N);

	fx = (float*)malloc(sizeof(float) * N);
	fy = (float*)malloc(sizeof(float) * N);
	ax = (float*)malloc(sizeof(float) * N);
	ay = (float*)malloc(sizeof(float) * N);
	num = (float*)malloc(sizeof(float) * D * D);

	//allocate GPU meomory for CUDA mode
	if (mode == CUDA) {
		cudaMalloc((void**) &d_x, sizeof(float) * N);
		cudaMalloc((void**) &d_y, sizeof(float) * N);
		cudaMalloc((void**) &d_vx, sizeof(float) * N);
		cudaMalloc((void**) &d_vy, sizeof(float) * N);
		cudaMalloc((void**) &d_m, sizeof(float) * N);

		cudaMalloc((void**)&(d_fx), sizeof(float) * N);
		cudaMalloc((void**)&(d_fy), sizeof(float) * N);
		cudaMalloc((void**)&(d_ax), sizeof(float) * N);
		cudaMalloc((void**)&(d_ay), sizeof(float) * N);
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
			printf("Error: Could not open input file. \n\n");
			exit(0);
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
							nbodies->x[j] = (float)atof(result);
							if (nbodies->x[j] <= EPSILON) nbodies->x[j] = (float)(rand()*1.0 / RAND_MAX);
							//printf("x: %f\n", nbodies[j].x);
							break;
						case 1:
							nbodies->y[j] = (float)atof(result);
							if (nbodies->y[j] <= EPSILON) nbodies->y[j] = (float)(rand()*1.0 / RAND_MAX);
							//printf("y: %f\n", nbodies[j].y);
							break;
						case 2:
							nbodies->vx[j] = (float)atof(result);
							if (nbodies->vx[j] <= EPSILON) nbodies->vx[j] = 0.0;
							//printf("vx: %f\n", nbodies[j].vx);
							break;
						case 3:
							nbodies->vy[j] = (float)atof(result);
							if (nbodies->vy[j] <= EPSILON) nbodies->vy[j] = 0.0;
							//printf("vy: %f\n", nbodies[j].vy);
							break;
						case 4:
							nbodies->m[j] = (float)atof(result);
							if (nbodies->m[j] <= EPSILON) nbodies->m[j] = (float)(1.0 / N);
							//printf("m: %f\n", nbodies[j].m);
							break;
						}
						//printf("result: %s\n", result);
						memset(result, 0, sizeof(result));
						ch2 = result;
						ch1++;
					}
				} 

				if (i != 5) {
					printf("Error: comma number should exactly be 4 in line %s\n", line);
				}
			}
			j++;
		}

		//the number of bodies in the input file is not N
		if (j != N) {
			printf("Error: The number of nbodies is wrong, change the value of N.\n\n");
			exit(0);
		}

		fclose(f);
	}
	// generate random data
	else {
		for (int i = 0; i < N; i++) {
			nbodies->x[i] = (float)(rand()*1.0 / RAND_MAX);
			nbodies->y[i] = (float)(rand()*1.0 / RAND_MAX);
			nbodies->vx[i] = 0.0;
			nbodies->vy[i] = 0.0;
			nbodies->m[i] = (float)(1.0 / N);
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
			printf("Error: wrong mode.\n");
			exit(0);
		}
		clock_t end_time = clock(); //record the Iteration ending time

		float run_time = (end_time - start_time) / (float) CLOCKS_PER_SEC;
		int seconds = (int) floor(run_time);
		int milliseconds = (int) ((run_time - seconds)*1000);
		printf("Execution time %d seconds %d milliseconds\n", seconds, milliseconds);
	}
	// visualisation mode 
	else {
		if (mode == CPU || mode == OPENMP) {
			initViewer(N, D, mode, step);
			setNBodyPositions2f(nbodies->x, nbodies->y);
			setActivityMapData(num);
			startVisualisationLoop();
		}
		else if (mode == CUDA) {

		}
		else {
			printf("Error: wrong mode.\n");
			exit(0);
		}
		
	}

	//free allocated memory
	free(nbodies->x);
	free(nbodies->y);
	free(nbodies->vx);
	free(nbodies->vy);
	free(nbodies->m);
	free(nbodies);
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
	memset(fx, 0, sizeof(float) * N);
	memset(fy, 0, sizeof(float) * N);
	memset(num, 0, sizeof(float) * D * D);

	//calculate this term firstly to avoid to calculate it redundantly
	float softening_powed = (float) (pow(SOFTENING, 2));

	//CPU mode 
	if (mode == CPU) {
		//calculate force and acceleration for all bodies
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				// calculate distance from body j to body i firstly  
				float d_x = nbodies->x[j] - nbodies->x[i];
				float d_y = nbodies->y[j] - nbodies->y[i];
				float distance_ij_powed = d_x * d_x + d_y * d_y;

				fx[i] += (float)(((double)nbodies->m[j] * d_x) / pow((double)distance_ij_powed + softening_powed, 3.0 / 2));
				fy[i] += (float)(((double)nbodies->m[j] * d_y) / pow((double)distance_ij_powed + softening_powed, 3.0 / 2));
				//printf("fx: %f, fy: %f\n", fx[i], fy[i]);
			}

			//calculate acceleration for body i
			ax[i] = fx[i] * G;
			ay[i] = fy[i] * G;
			//printf("ax: %f, ay: %f\n", ax[i], ay[i]);
		}

		//calculate motion for all bodies 
		for (int i = 0; i < N; i++) {
			nbodies->vx[i] = nbodies->vx[i] + dt * ax[i];
			nbodies->vy[i] = nbodies->vy[i] + dt * ay[i];
			//printf("vx: %f, vy: %f\n", nbodies[i].vx, nbodies[i].vy);

			//calculate the location for body i
			nbodies->x[i] = nbodies->x[i] + dt * nbodies->vx[i];
			nbodies->y[i] = nbodies->y[i] + dt * nbodies->vy[i];
			//printf("x: %f, y: %f\n", nbodies[i].x, nbodies[i].y);
		}

		//calculate activity map
		//get the location of body i in the one dimensional array with D*D length
		memset(num, 0, sizeof(float) * D * D);
		for (int i = 0; i < N; i++) {
			int row = (int) floor((double)nbodies->x[i] * D);
			int column = (int) floor((double)nbodies->y[i] * D);

			if (row >= D)	row = D - 1;
			if (row <= 0)	row = 0;
			if (column >= D)	column = D - 1;
			if (column <= 0)	column = 0;
			num[column + row * D] += (float) (1.0*D / N);
		}
		/*
		for (int i = 0; i < D; i++) {
			for (int j = 0; j < D; j++) {
				printf("%f, ", num[i*D+j]);
			}
			printf("\n");
		}*/
	}
	//OPENMP mode 
	else if (mode == OPENMP) {
		//calculate force and acceleration for all bodies
		int i, j;
		for (i = 0; i < N; i++) {
			#pragma omp parallel for shared(nbodies)
			for (j = 0; j < N; j++) {
				// calculate distance from body j to body i firstly  
				float d_x = nbodies->x[j] - nbodies->x[i];
				float d_y = nbodies->y[j] - nbodies->y[i];
				float distance_ij_powed = d_x * d_x + d_y * d_y;

				fx[i] += (float)(((double)nbodies->m[j] * d_x) / pow((double)distance_ij_powed + softening_powed, 3 / 2));
				fy[i] += (float)(((double)nbodies->m[j] * d_y) / pow((double)distance_ij_powed + softening_powed, 3 / 2));
			}

			//calculate acceleration for body i
			ax[i] = fx[i] * G;
			ay[i] = fy[i] * G;
			#pragma omp barrier
		}
		

		//calculate motion for all bodies 
		#pragma omp parallel for shared(nbodies,N,ax,ay) private(i)
		for (i = 0; i < N; i++) {
			nbodies->vx[i] = nbodies->vx[i] + dt * ax[i];
			nbodies->vy[i] = nbodies->vy[i] + dt * ay[i];
			//printf("vx: %f, vy: %f\n", nbodies[i].vx, nbodies[i].vy);

			//calculate the location for body i
			nbodies->x[i] = nbodies->x[i] + dt * nbodies->vx[i];
			nbodies->y[i] = nbodies->y[i] + dt * nbodies->vy[i];
		}
		#pragma omp barrier

		//calculate activity map
		//get the location of body i in the one dimensional array with D*D length
		memset(num, 0, sizeof(float) * D * D);

		#pragma omp parallel for shared(nbodies,N,D,num) private(i)
		for (i = 0; i < N; i++) {
			int row = (int)floor((double)nbodies->x[i] * D);
			int column = (int)floor((double)nbodies->y[i] * D);

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
		printf("Error: wrong mode.\n");
		exit(0);
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
