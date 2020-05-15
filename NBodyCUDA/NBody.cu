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

void print_help();
void step(void);

//define and initialize global variables
int N = 0;
int D = 0;
MODE mode;
int I = 0;
char *input_file = NULL;
nbody *nbodies = NULL;
float *fx = NULL, *fy = NULL, *ax = NULL, *ay = NULL, *num = NULL;

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

	nbodies = (struct nbody*)malloc(sizeof(struct nbody) * N);
	fx = (float*)malloc(sizeof(float) * N);
	fy = (float*)malloc(sizeof(float) * N);
	ax = (float*)malloc(sizeof(float) * N);
	ay = (float*)malloc(sizeof(float) * N);
	num = (float*)malloc(sizeof(float) * D * D);
	
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

			if (j <= N) {
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
							nbodies[j].x = (float)atof(result);
							if (nbodies[j].x <= EPSILON) nbodies[j].x = (float)(rand()*1.0 / RAND_MAX);
							//printf("x: %f\n", nbodies[j].x);
							break;
						case 1:
							nbodies[j].y = (float)atof(result);
							if (nbodies[j].y <= EPSILON) nbodies[j].y = (float)(rand()*1.0 / RAND_MAX);
							//printf("y: %f\n", nbodies[j].y);
							break;
						case 2:
							nbodies[j].vx = (float)atof(result);
							if (nbodies[j].vx <= EPSILON) nbodies[j].vx = 0.0;
							//printf("vx: %f\n", nbodies[j].vx);
							break;
						case 3:
							nbodies[j].vy = (float)atof(result);
							if (nbodies[j].vy <= EPSILON) nbodies[j].vy = 0.0;
							//printf("vy: %f\n", nbodies[j].vy);
							break;
						case 4:
							nbodies[j].m = (float)atof(result);
							if (nbodies[j].m <= EPSILON) nbodies[j].m = (float)(1.0 / N);
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
 
				j++;
			}
			//the number of bodies in the input file is larger than N
			else {
				printf("Error: The number of nbodies is wrong, increase the value of N.\n\n");
				exit(0);
			}
		}

		//the number of bodies in the input file is less tha N
		if (j < N) {
			printf("Error: The number of nbodies is wrong, decrease the value of N.\n\n");
			exit(0);
		}

		fclose(f);
	}
	// generate random data
	else {
		for (int i = 0; i < N; i++) {
			nbodies[i].x = (float)(rand()*1.0 / RAND_MAX);
			nbodies[i].y = (float)(rand()*1.0 / RAND_MAX);
			nbodies[i].vx = 0.0;
			nbodies[i].vy = 0.0;
			nbodies[i].m = (float)(1.0 / N);
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
		initViewer(N, D, mode, step);
		setNBodyPositions(nbodies);
		setActivityMapData(num);
		startVisualisationLoop();
	}

	//free allocated memory
	free(nbodies);
	free(fx);
	free(fy);
	free(ax);
	free(ay);
	free(num);

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
				float d_x = nbodies[j].x - nbodies[i].x;
				float d_y = nbodies[j].y - nbodies[i].y;
				float distance_ij_powed = d_x * d_x + d_y * d_y;

				fx[i] += (float)(((double) nbodies[j].m * d_x) / pow((double)distance_ij_powed + softening_powed, 3 / 2));
				fy[i] += (float)(((double)nbodies[j].m * d_y) / pow((double)distance_ij_powed + softening_powed, 3 / 2));
				//printf("fx: %f, fy: %f\n", fx[i], fy[i]);
			}

			//multiply G and mass for body i's Force
			fx[i] *= G * nbodies[i].m;
			fy[i] *= G * nbodies[i].m;
			//printf("fx: %f, fy: %f\n", fx[i], fy[i]);

			//calculate acceleration for body i
			ax[i] = fx[i] / nbodies[i].m;
			ay[i] = fy[i] / nbodies[i].m;
			//printf("ax: %f, ay: %f\n", ax[i], ay[i]);
		}

		//calculate motion for all bodies 
		for (int i = 0; i < N; i++) {
			nbodies[i].vx = nbodies[i].vx + dt * ax[i];
			nbodies[i].vy = nbodies[i].vy + dt * ay[i];
			//printf("vx: %f, vy: %f\n", nbodies[i].vx, nbodies[i].vy);

			//calculate the location for body i
			nbodies[i].x = nbodies[i].x + dt * nbodies[i].vx;
			nbodies[i].y = nbodies[i].y + dt * nbodies[i].vy;
			//printf("x: %f, y: %f\n", nbodies[i].x, nbodies[i].y);
		}

		//calculate activity map
		//get the location of body i in the one dimensional array with D*D length
		memset(num, 0, sizeof(float) * D * D);
		for (int i = 0; i < N; i++) {
			int row = (int) floor((double)nbodies[i].x * D);
			int column = (int) floor((double)nbodies[i].y * D);

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
		#pragma omp parallel for shared(nbodies,N,fx,fy,ax,ay)
		for (i = 0; i < N; i++) {
			for (j = 0; j < N; j++) {
				// calculate distance from body j to body i firstly  
				float d_x = nbodies[j].x - nbodies[i].x;
				float d_y = nbodies[j].y - nbodies[i].y;
				float distance_ij_powed = d_x * d_x + d_y * d_y;

				fx[i] += (float)(((double)nbodies[j].m * d_x) / pow((double)distance_ij_powed + softening_powed, 3 / 2));
				fy[i] += (float)(((double)nbodies[j].m * d_y) / pow((double)distance_ij_powed + softening_powed, 3 / 2));
			}

			//multiply G and mass for body i's Force
			fx[i] *= G * nbodies[i].m;
			fy[i] *= G * nbodies[i].m;

			//calculate acceleration for body i
			ax[i] = fx[i] / nbodies[i].m;
			ay[i] = fy[i] / nbodies[i].m;
		}
		#pragma omp barrier

		//calculate motion for all bodies 
		#pragma omp parallel for default(none) shared(nbodies,N,ax,ay) private(i)
		for (i = 0; i < N; i++) {
			nbodies[i].vx = nbodies[i].vx + dt * ax[i];
			nbodies[i].vy = nbodies[i].vy + dt * ay[i];

			//calculate the location for body i
			nbodies[i].x = nbodies[i].x + dt * nbodies[i].vx;
			nbodies[i].y = nbodies[i].y + dt * nbodies[i].vy;
		}
		#pragma omp barrier

		//calculate activity map
		//get the location of body i in the one dimensional array with D*D length
		memset(num, 0, sizeof(float) * D * D);

		#pragma omp parallel for shared(nbodies,N,D,num) private(i)
		for (i = 0; i < N; i++) {
			int row = (int)floor((double)nbodies[i].x * D);
			int column = (int)floor((double)nbodies[i].y * D);

			if (row >= D || row <= 0)	continue;
			if (column >= D || column <= 0)	continue;
			#pragma omp atomic
			num[column + row * D] += (float)(1.0*D / N);
		}
	}
	//CUDA mode
	else if (mode == CUDA) {
	
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