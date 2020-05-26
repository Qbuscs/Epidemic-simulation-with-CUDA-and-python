#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<curand.h>
#include<curand_kernel.h>
#include<string.h>
#include<new>

#define FALSE 0
#define TRUE 1
#define STR_EQ 0

#define max(a, b) \
	({__typeof__ (a) _a = (a); \
		__typeof__ (b) _b = (b); \
		_a > _b ? _a : _b; })
		
#define min(a, b) \
	({__typeof__ (a) _a = (a); \
		__typeof__ (b) _b = (b); \
		_a > _b ? _b : _a; })
		
#define abs(a) \
	({__typeof__ (a) _a = (a); \
		_a >= 0 ? _a : -_a; })

/* =================== BASIC FUNCTIONS =====================================================================*/
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__device__ void curandInit(curandState_t* state_ptr, int tid){
	curand_init((unsigned long long)clock(), tid, 0, state_ptr);
}

__device__ float cudaFloatRand(float min, float max, curandState_t* state_ptr){
	return min + curand_uniform(state_ptr) * (max - min);
}

__device__ int cudaIntRand(int min, int max, curandState_t* state_ptr){
	return int(cudaFloatRand(float(min), float(max + 1.0), state_ptr));
}

__host__ float floatRand(float min, float max){
	float scale = rand() / (float) RAND_MAX;
	return min + scale * (max - min);
}	

__host__ char roll(float probability){
	if(floatRand(0.0, 1.0) < probability)
		return TRUE;
	return FALSE;
}	
__device__ char cudaRoll(float probability, curandState_t* curand_state_ptr){
	if(cudaFloatRand(0.0, 1.0, curand_state_ptr) < probability)
		return TRUE;
	return FALSE;
}

/* =================== STRUCTS AND METHODS =====================================================================*/

typedef struct SimulationOptions{
	int N;
	float DIM;
	int simulation_time;
	float infection_r;
	float infection_p;
	float velocity;
	int immune_time;
	int sympthoms_time;
	int blocks;
	int threads_per_block;
	char* output;
	float lawful_p;
	int quarantine_sick_time;
	int quarantine_all_time;
	int gathering_points_n;
	float gathering_point_p;
	int buffor_size;
} SimulationOptions;

typedef enum{HEALTHY, CARRIER, SICK, IMMUNE} Health;
typedef enum{GOING_TO, GOING_BACK, NO_DESTINATION} GatheringPointTravel;

typedef struct Point{
	float x;
	float y;
} Point;

__host__ Point randPoint(float DIM){
	Point point;
	point.x = floatRand(0.0, DIM);
	point.y = floatRand(0.0, DIM);
	return point;
}

__device__ Point cudaRandPoint(float DIM, curandState_t* state_ptr){
	Point point;
	point.x = cudaFloatRand(0.0, DIM, state_ptr);
	point.y = cudaFloatRand(0.0, DIM, state_ptr);
	return point;
}

__host__ __device__ float distance(Point p1, Point p2){
	float dx = abs(p1.x - p2.x);
	float dy = abs(p1.y - p2.y);
	return sqrt(dx * dx + dy * dy);
}

typedef struct Person{
	Point location;
	Point home;
	Health health;
	GatheringPointTravel travel;
	char quarantined; // SICK people are totaly quarantined, the rest is partialy quarantined
	int time_sick;
	Point destination;
	char lawful;
} Person;

typedef struct PersonInfo{
	Point location;
	Health health;
} PersonInfo;

/* =================== DEVICE CODE =====================================================================*/

__device__ void updateQuarantine(SimulationOptions settings, Person* person_ptr, int time){
	if(!(person_ptr->lawful))
		return;
	if(settings.quarantine_all_time && settings.quarantine_all_time < time)
		person_ptr->quarantined = TRUE;
	else if(settings.quarantine_sick_time && settings.quarantine_sick_time < time){
		if(person_ptr->health == SICK){
			person_ptr->quarantined = TRUE;
			person_ptr->travel = NO_DESTINATION;
		}
		else
			person_ptr->quarantined = FALSE;
	}
}

__device__ void migrate(
	SimulationOptions settings,
	Person* person_ptr,
	curandState_t* state_ptr,
	Point* gathering_points
){
	float angle, dy, dx;
	float destination_r = settings.velocity;
	
	if(person_ptr->quarantined){
		if(person_ptr->health == SICK)
			return;
		if(person_ptr->travel == GOING_TO && distance(person_ptr->location, person_ptr->destination) < destination_r){
			person_ptr->destination = person_ptr->home;
			person_ptr->travel = GOING_BACK;
		}
		if(person_ptr->travel == GOING_BACK && distance(person_ptr->location, person_ptr->destination) < destination_r){
			person_ptr->travel = NO_DESTINATION;
		}
		if(person_ptr->travel == NO_DESTINATION){
			if(!settings.gathering_points_n)
				return;
			if(!cudaRoll(settings.gathering_point_p, state_ptr))
				return;
			person_ptr->destination = gathering_points[cudaIntRand(0, settings.gathering_points_n - 1, state_ptr)];
			person_ptr->travel = GOING_TO;
		}
	}
	
	else if(distance(person_ptr->location, person_ptr->destination) < destination_r){
		person_ptr->destination = cudaRandPoint(settings.DIM, state_ptr);
	}
	
	dy = person_ptr->destination.y - person_ptr->location.y;
	dx = person_ptr->destination.x - person_ptr->location.x;
	angle = atan2(dy, dx);
	person_ptr->location.x = min(max(person_ptr->location.x + cos(angle) * settings.velocity, 0.0), settings.DIM);
	person_ptr->location.y = min(max(person_ptr->location.y + sin(angle) * settings.velocity, 0.0), settings.DIM);
}

__device__ void developDisease(SimulationOptions settings, Person* person_ptr){
	if(person_ptr->health == CARRIER || person_ptr->health == SICK)
		person_ptr->time_sick += 1;
	if(person_ptr->time_sick > settings.immune_time)
		person_ptr->health = IMMUNE;
	else if(person_ptr->time_sick > settings.sympthoms_time)
		person_ptr->health = SICK;
}

// there may be races, but it doesn't matter (I think?)
__device__ void infect(
	SimulationOptions settings,
	Person* population,
	int me_idx,
	curandState_t* curand_state_ptr
){
	Person* me_ptr = &population[me_idx];
	Person* person_ptr;
	int i;
	if((me_ptr->health == CARRIER || me_ptr->health == SICK) && !(me_ptr->quarantined && me_ptr->health == SICK)){
		for(i = 0; i < settings.N; i++){
			person_ptr = &population[i];
			if(i == me_idx) continue;
			if(person_ptr->quarantined && person_ptr->travel == NO_DESTINATION) continue;
			if(person_ptr->health == CARRIER || person_ptr->health == SICK) continue;
			if(distance(me_ptr->location, person_ptr->location) > settings.infection_r) continue;
			if(cudaRoll(settings.infection_p, curand_state_ptr))
				person_ptr->health = CARRIER;
		}
	}
}

__global__ void simulate(
	SimulationOptions settings,
	Person* population,
	curandState_t* curand_states,
	int time,
	Point* gathering_points,
	int buffor_index,
	PersonInfo* population_info
){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int i;
	Person* person_ptr;
	curandState_t my_curand_state = curand_states[tid];
	curandInit(&my_curand_state, tid);
	
	// develop disease
	i = tid;
	while(i < settings.N){
		person_ptr = &population[i];
		developDisease(settings, person_ptr);
		i += gridDim.x * blockDim.x;
	}
	
	// update population quarantine_all_time
	i = tid;
	while(i < settings.N){
		person_ptr = &population[i];
		updateQuarantine(settings, person_ptr, time);
		i += gridDim.x * blockDim.x;
	}
	
	// migration of population
	i = tid;
	while(i < settings.N){
		person_ptr = &population[i];
		migrate(settings, person_ptr, &my_curand_state, gathering_points);
		i += gridDim.x * blockDim.x;
	}
	
	// spread of disease
	i = tid;
	while(i < settings.N){
		infect(settings, population, i, &my_curand_state);
		i += gridDim.x * blockDim.x;
	}
	
	// save to buffor
	i = tid;
	while(i < settings.N){
		population_info[settings.N * buffor_index + i].location = population[i].location;
		population_info[settings.N * buffor_index + i].health = population[i].health;
		i += gridDim.x * blockDim.x;
	}
}

/* =================== HOST =====================================================================*/

int main(int argc, char** argv){
	SimulationOptions settings;
	int i, j, buffors_simulated;
	FILE* file;
	char save_output;
	Person* population;
	Person* dev_population;
	curandState_t* curand_states;
	Point* gathering_points;
	Point* dev_gathering_points;
	PersonInfo* population_info;
	PersonInfo* dev_population_info;
	
	settings.N = 10000;
	settings.DIM = 100;
	settings.simulation_time = 500;
	settings.velocity = 1.0;
	settings.infection_p = 0.33;
	settings.infection_r = 3.0;
	settings.immune_time = 100;
	settings.sympthoms_time = 10;
	settings.blocks = 128;
	settings.threads_per_block = 128;
	settings.output = "output.sim";
	settings.quarantine_all_time = 0;
	settings.quarantine_sick_time = 0;
	settings.lawful_p = 1.0;
	settings.gathering_points_n = 0;
	settings.gathering_point_p = 0.05;
	settings.buffor_size = 1;
	
	//read commandline args
	i = 1;
	while(i < argc - 1){
		if(strcmp(argv[i], "--N") == STR_EQ || strcmp(argv[i], "-N") == STR_EQ){
			settings.N = atoi(argv[++i]);
			if(settings.N < 1) return 1;
		}
		else if(strcmp(argv[i], "-=DIM") == STR_EQ || strcmp(argv[i], "-DIM") == STR_EQ){
			settings.DIM = atof(argv[++i]);
			if(settings.DIM <= 0.0) return 1;
		}
		else if(strcmp(argv[i], "--simulation_n") == STR_EQ || strcmp(argv[i], "-simn") == STR_EQ){
			settings.simulation_time = atoi(argv[++i]);
			if(settings.simulation_time < 1) return 1;
		}
		else if(strcmp(argv[i], "--velocity") == STR_EQ || strcmp(argv[i], "-v") == STR_EQ){
			settings.velocity = atof(argv[++i]);
			if(settings.velocity < 0) return 1;
		}
		else if(strcmp(argv[i], "--infection_p") == STR_EQ || strcmp(argv[i], "-infp") == STR_EQ){
			settings.infection_p = atof(argv[++i]);
			if(settings.infection_p <= 0.0) return 1;
		}
		else if(strcmp(argv[i], "--infection_r") == STR_EQ || strcmp(argv[i], "-infr") == STR_EQ){
			settings.infection_r = atof(argv[++i]);
			if(settings.infection_r <= 0.0) return 1;
		}
		else if(strcmp(argv[i], "--immune_time") == STR_EQ || strcmp(argv[i], "-immt") == STR_EQ){
			settings.immune_time = atoi(argv[++i]);
			if(settings.immune_time < 0) return 1;
		}
		else if(strcmp(argv[i], "--sympthoms_time") == STR_EQ || strcmp(argv[i], "-symt") == STR_EQ){
			settings.sympthoms_time = atoi(argv[++i]);
			if(settings.sympthoms_time < 0) return 1;
		}
		else if(strcmp(argv[i], "--blocks") == STR_EQ || strcmp(argv[i], "-b") == STR_EQ){
			settings.blocks = atoi(argv[++i]);
			if(settings.blocks < 1) return 1;
		}
		else if(strcmp(argv[i], "--threads_per_block") == STR_EQ || strcmp(argv[i], "-tpb") == STR_EQ){
			settings.threads_per_block = atoi(argv[++i]);
			if(settings.threads_per_block < 1) return 1;
		}
		else if(strcmp(argv[i], "--output") == STR_EQ || strcmp(argv[i], "-o") == STR_EQ){
			settings.output = argv[++i];
			if(!settings.output) return 1;
		}
		else if(strcmp(argv[i], "--quarantine_all_time") == STR_EQ || strcmp(argv[i], "-qat") == STR_EQ){
			settings.quarantine_all_time = atoi(argv[++i]);
			if(settings.quarantine_all_time < 0) return 1;
		}
		else if(strcmp(argv[i], "--quarantine_sick_time") == STR_EQ || strcmp(argv[i], "-qst") == STR_EQ){
			settings.quarantine_sick_time = atoi(argv[++i]);
			if(settings.quarantine_sick_time < 0) return 1;
		}
		else if(strcmp(argv[i], "--lawful_p") == STR_EQ || strcmp(argv[i], "-lawp") == STR_EQ){
			settings.lawful_p = atof(argv[++i]);
			if(settings.lawful_p < 0.0) return 1;
		}
		else if(strcmp(argv[i], "--gathering_points_n") == STR_EQ || strcmp(argv[i], "-gn") == STR_EQ){
			settings.gathering_points_n = atoi(argv[++i]);
			if(settings.gathering_points_n < 0) return 1;
		}
		else if(strcmp(argv[i], "--gathering_point_p") == STR_EQ || strcmp(argv[i], "-gp") == STR_EQ){
			settings.gathering_point_p = atof(argv[++i]);
			if(settings.gathering_point_p < 0.0) return 1;
		}
		else if(strcmp(argv[i], "--buffor_size") == STR_EQ || strcmp(argv[i], "-buff") == STR_EQ){
			settings.buffor_size = atoi(argv[++i]);
			if(settings.buffor_size < 1) return 1;
		}
		i++;
	}
	
	
	
	if(strcmp(settings.output, "none") == STR_EQ)
		save_output = FALSE;
	else
		save_output = TRUE;
	
	try{
		population_info = new PersonInfo[settings.N * settings.buffor_size];
		population = new Person[settings.N];
	}
	catch(const std::bad_alloc& e){
		printf("Insufficent memory on host\n");
		return 1;
	}
	
	
	srand((unsigned int)time(NULL));
	
	for(i = 0; i < settings.N; i++){
		population[i].location.x = floatRand(0.0, settings.DIM);
		population[i].location.y = floatRand(0.0, settings.DIM);
		population[i].home = population[i].location;
		population[i].destination.x = floatRand(0.0, settings.DIM);
		population[i].destination.y = floatRand(0.0, settings.DIM);
		population[i].health = HEALTHY;
		population[i].quarantined = FALSE;
		population[i].time_sick = 0;
		population[i].travel = NO_DESTINATION;
		if(roll(settings.lawful_p))
			population[i].lawful = TRUE;
		else
			population[i].lawful = FALSE;
	}
	
	gathering_points = new Point[settings.gathering_points_n];
	for(i = 0; i < settings.gathering_points_n; i++){
		gathering_points[i].x = floatRand(0.0, settings.DIM);
		gathering_points[i].y = floatRand(0.0, settings.DIM);
	}
	
	
	//patient zero
	population[0].health = CARRIER;
	
	HANDLE_ERROR( cudaMalloc((void**)&dev_population, sizeof(Person) * settings.N) );
	HANDLE_ERROR( cudaMalloc((void**)&curand_states, sizeof(curandState_t) * settings.blocks * settings.threads_per_block) );
	HANDLE_ERROR( cudaMalloc((void**)&dev_gathering_points, sizeof(Point) * settings.gathering_points_n) );
	HANDLE_ERROR( cudaMalloc((void**)&dev_population_info, sizeof(PersonInfo) * settings.N * settings.buffor_size) );
	
	HANDLE_ERROR( cudaMemcpy(dev_population, population, sizeof(Person) * settings.N, cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(dev_gathering_points, gathering_points, sizeof(Point) * settings.gathering_points_n, cudaMemcpyHostToDevice) );

	if(save_output){
		file = fopen(settings.output, "w");
		fprintf(file, "%d %f %d %d\n", settings.N, settings.DIM, settings.simulation_time, settings.gathering_points_n);
		for(i = 0; i < settings.gathering_points_n; i++)
			fprintf(file, "%f %f\n", gathering_points[i].x, gathering_points[i].y);
	}
	// for(i = 0; i < settings.simulation_time; i++){
		// printf("==========SIM%d==========\n", i);
		// simulate<<<settings.blocks, settings.threads_per_block>>>(settings, dev_population, curand_states, i, dev_gathering_points);
		// cudaDeviceSynchronize();
		// HANDLE_ERROR( cudaMemcpy(population, dev_population, sizeof(Person) * settings.N, cudaMemcpyDeviceToHost) );
		// if(save_output){
			// for(j = 0; j < settings.N; j++){
				// fprintf(file, "%f %f %d\n", population[j].location.x, population[j].location.y, population[j].health);
			// }
		// }
	// }
	i = 0;
	while(i < settings.simulation_time){
		for(j = 0; j < settings.buffor_size; j++){
			printf("==========SIM%d==========\n", i);
			simulate<<<settings.blocks, settings.threads_per_block>>>(
				settings, dev_population, curand_states, i, dev_gathering_points, j, dev_population_info
			);
			cudaDeviceSynchronize();
			buffors_simulated = j + 1;
			i++;
			if(i >= settings.simulation_time)
				break;
		}
		printf("Coping buffor from GPU to host...\n");
		HANDLE_ERROR( cudaMemcpy(
			population_info, dev_population_info, sizeof(PersonInfo) * settings.N * settings.buffor_size, cudaMemcpyDeviceToHost
		) );
		if(save_output){
			for(j = 0; j < settings.N * buffors_simulated; j++){
				fprintf(file, "%f %f %d\n", population_info[j].location.x, population_info[j].location.y, population_info[j].health);
			}
		}
	}
	
	if(save_output)
		fclose(file);
	cudaFree(curand_states);
	cudaFree(dev_population);
	cudaFree(dev_gathering_points);
	cudaFree(dev_population_info);
	delete[] population;
	delete[] gathering_points;
	delete[] population_info;
	return 0;
}