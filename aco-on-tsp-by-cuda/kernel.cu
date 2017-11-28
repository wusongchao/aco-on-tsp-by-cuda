#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <cfloat>
#include <string>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <random>
#include <map>
#include <algorithm>
#include <fstream>

#define CUDA_CALL(x) {const cudaError_t a = (x);if(a!=cudaSuccess){printf("\nCUDA Error:%s(err_num=%d)\n",cudaGetErrorString(a),a);}}

typedef struct Point {
	int index;
	double x;
	double y;
}Point;

__device__ void warpReduce(int tid, float in, float* data)
{
	int idx = (2 * tid - (tid & 0x1f));
	data[idx] = 0;
	idx += 32;
	float t = data[idx] = in;

	data[idx] = t = t + data[idx - 1];
	data[idx] = t = t + data[idx - 2];
	data[idx] = t = t + data[idx - 4];
	data[idx] = t = t + data[idx - 8];
	data[idx] = t = t + data[idx - 16];
}

constexpr int MAX_POINT_NUM = 256;

void clearAllowed(bool* allowed, int cityNum)
{
	for (int i = 1;i <= cityNum;i++) {
		allowed[i] = true;
	}
}

int calculateProbabilityAndSelect(double Tau[][MAX_POINT_NUM], double heuristicValue[][MAX_POINT_NUM],
	bool allowed[], int currentCityIndex, int cityNum, double alpha, double beta)
{
	std::map<int, double> probability;

	double denominatorSum = 0;
	for (int s = 1;s <= cityNum;s++) {
		if (allowed[s] == true) {
			double val = pow(Tau[currentCityIndex][s], alpha)*pow(heuristicValue[currentCityIndex][s], beta);
			probability.insert({ s,val });
			denominatorSum += val;
		}
	}

	int selectedCityIndex = -1;
	if (probability.size() == 1) {
		//means we have no city for choosing
		selectedCityIndex = probability.begin()->first;
	}
	else {
		double sumSelect = 0;

		std::default_random_engine randomEngine;
		std::uniform_real_distribution<double> randomDistribution(0, 1);
		double generatedProbablity = randomDistribution(randomEngine);

		for (auto &r : probability) {
			r.second = r.second / denominatorSum;
			sumSelect += r.second;
			if (sumSelect >= generatedProbablity) {
				selectedCityIndex = r.first;
				break;
			}
		}
		selectedCityIndex = selectedCityIndex == -1 ? (--probability.end())->first : selectedCityIndex;
	}

	allowed[selectedCityIndex] = false;

	return selectedCityIndex;
}

void calculateDeltaTauPerAnt(double deltaTau[][MAX_POINT_NUM], int Q, std::vector<int>* path, double antPathLength)
{
	for (int k = 0, pathLength = path->size() - 1;k < pathLength;k++) {
		int i = path->operator[](k);
		int j = path->operator[](k + 1);
		deltaTau[i][j] = deltaTau[j][i] += Q / antPathLength;
	}
}


void updateTau(double Tau[][MAX_POINT_NUM], double deltaTauTotal[][MAX_POINT_NUM], double rho, int cityNum)
{
	for (int i = 1;i <= cityNum;i++) {
		for (int j = 1;j <= cityNum;j++) {
			double val = (1.0 - rho) * Tau[i][j] + deltaTauTotal[i][j];
			Tau[j][i] = Tau[i][j] = val;
		}
	}
}

__inline__ __device__ void clearTauMatrix(unsigned int tid, float* src)
{
	src[tid] = 0.0f;
}

__inline__ __device__ void extractTSPData(int tid, int firstDimension, int secondDimension, struct Point* points, float* graphMatrix, float* heuristicValue, int cityNum)
{
	if (firstDimension != secondDimension && firstDimension < cityNum && secondDimension < cityNum) {
		struct Point lhs = points[firstDimension];
		struct Point rhs = points[secondDimension];
		float val = sqrtf((lhs.x - rhs.x)*(lhs.x - rhs.x) + (lhs.y - rhs.y)*(lhs.y - rhs.y));
		graphMatrix[tid] = val;
		heuristicValue[tid] = 1 / val;
	}
}

__inline__ __device__ void initializeTauState(int tid, float* Tau, float* deltaTau)
{
	Tau[tid] = 1.0f;
	deltaTau[tid] = 0.0f;
}

__global__ void clearTabu(bool* Tabu)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	Tabu[tid] = true;
}

__global__ void preProcessing(struct Point* points, float* graphMatrix, float* heuristicValue, float* Tau, float* deltaTau, int cityNum)
{
	// to reduce memory request
	int firstDimension = blockIdx.x;
	int secondDimension = threadIdx.x;
	int tid = firstDimension * MAX_POINT_NUM + secondDimension;
	extractTSPData(tid, firstDimension, secondDimension, points, graphMatrix, heuristicValue, cityNum);
	initializeTauState(tid, Tau, deltaTau);
}

// for each block(ant), we have only one "randomEngine"
__inline__ __device__ int generateInitialCityIndex(curandState* state, int cityNum)
{
	// since city index in array will begin with 0
	return curand(state) % cityNum;
}

//__shared__ int path[MAX_POINT_NUM];
//__shared__ int current;
//__shared__ int startCityIndex;
//__shared__ int pathSize;
//__shared__ float currentDistance;

//__device__ void calculateProbability(int tid, float* Tau, float* heuristicValue, int cityNum, float alpha, float beta)
//{
//	// each thread mapped to a city(start from zero)
//	if (allowed[tid] == true) {
//		const int offset = MAX_POINT_NUM;
//		// must ensure that val not all is 0, so, have to make Tau an initial value instead of 0
//		float val = powf(Tau[current * offset + tid], alpha)*powf(heuristicValue[current * offset + tid], beta) + 0.1f;
//		// means possibility from current to tid
//		probabilities[tid] = val;
//		atomicAdd(&denominatorSum, val);
//	}
//}
//
//__device__ int selectCity(curandState* state, int cityNum)
//{
//	float generatedProbablity = curand_uniform(state);
//	float sumSelect = 0.0f;
//	int selectedIndex = 0;
//	for (int i = 0; i < cityNum;i++) {
//		if (allowed[i] == true) {
//			//sumSelect += 1;
//			sumSelect += (probabilities[i] / denominatorSum);
//			if (sumSelect >= generatedProbablity) {
//				selectedIndex = i;
//				break;
//			}
//		}
//	}
//	return selectedIndex;
//}

__global__ void constructPath(float* Tau, float* deltaTau, float* graphMatrix, float* heuristicValue, curandState* states, int cityNum, float alpha, float beta, int Q, 
	float* bestLength, int* bestPath)
{
	__shared__ int path[MAX_POINT_NUM];
	__shared__ int current;
	__shared__ int startCityIndex;
	//__shared__ int pathSize;
	__shared__ float currentDistance;
	__shared__ float probabilities[MAX_POINT_NUM];
	__shared__ float denominatorSum;
	__shared__ bool allowed[MAX_POINT_NUM];

	int tid = threadIdx.x;
	int bid = blockIdx.x;
	const int offset = MAX_POINT_NUM;
	
	if (tid < cityNum) {
		int innerPathSize = 0;
		allowed[tid] = true;
		if (tid == 0) {
			int temp = generateInitialCityIndex(states + bid, cityNum);
			path[innerPathSize] = current = startCityIndex = temp;
			currentDistance = 0.0f;
			allowed[current] = false;
			denominatorSum = 0.0f;
		}
		innerPathSize = 1;
		__syncthreads();

		while (innerPathSize < cityNum) {
			////if (tid < cityNum) {
			//	calculateProbability(tid, Tau, heuristicValue, cityNum, alpha, beta);
			//	// must ensure that before select, the probabilities have been calculated
			////}
			if (allowed[tid] == true) {
				const int offset = MAX_POINT_NUM;
				// must ensure that val not all is 0, so, have to make Tau an initial value instead of 0
				float val = powf(Tau[current * offset + tid], alpha)*powf(heuristicValue[current * offset + tid], beta);
				// means possibility from current to tid
				probabilities[tid] = val;
				atomicAdd(&denominatorSum, val);
			}

			__syncthreads();

			if (tid == 0) {
				float generatedProbablity = curand_uniform(states + bid);
				float sumSelect = 0.0f;
				int selectedIndex = 0;
				for (int i = 0; i < cityNum;i++) {
					if (allowed[i] == true) {
						//sumSelect += 1.0f;
						sumSelect += (probabilities[i] / denominatorSum);
						if (sumSelect >= generatedProbablity) {
							selectedIndex = i;
							break;
						}
					}
				}
				int next = selectedIndex;
				//int next = selectCity(states + bid, cityNum);
				allowed[next] = false;
				currentDistance += graphMatrix[current * offset + next];
				path[innerPathSize] = next;
				current = next;
				denominatorSum = 0;
			}
			innerPathSize++;
			__syncthreads();
		}

		if (tid == 0) {
			currentDistance += graphMatrix[current * offset + startCityIndex];
			float bestLengthForThisAnt = bestLength[bid];
			if (currentDistance < bestLengthForThisAnt) {
				bestLength[bid] = currentDistance;
				int off = bid * offset;
				for (int i = 0; i < cityNum;i++) {
					bestPath[off + i] = path[i];
				}
			}
		}
		else {
			float val = Q / currentDistance;
			atomicAdd(&deltaTau[path[tid - 1] * offset + path[tid]], val);
			atomicAdd(&deltaTau[path[tid] * offset + path[tid - 1]], val);
		}
	}
}

__global__ void updatePheromones(float* Tau, float* deltaTauTotal, double rho, int cityNum)
{
	// blockIdx.x is the first dimension
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float tau = Tau[tid];
	tau = (1 - rho) * tau;
	tau += deltaTauTotal[tid];

	// clear the deltaTau
	deltaTauTotal[tid] = 0.0f;
	Tau[tid] = tau;
}

__global__ void setupCuRandKernel(curandState* state)
{
	
    #define THREAD_SEED 1234
	int id = threadIdx.x;
	curand_init(THREAD_SEED, id, 0, state + id);
}

__global__ void initializeEachAntsBestLength(float* length)
{
	length[threadIdx.x] = FLT_MAX;
}

// use a sub-blocks to represent the range
// that is, if we have m city left in the unvisited list, we have m sub-blocks
// than use a __ballot(probability in range?) operation to find out which sub-block "win"
//unsigned int __ballot(int x)
//{
//	if (x != 0) {
//		return (1 << (threadIdx.x % 32));
//	}
//	return 0;
//}

int main()
{
	constexpr int antNum = 32;
	constexpr int iterTimes = 600;
	double alpha = 1;//信息素的重要程度
	double beta = 3.8;//启发式因子的重要程度
	double rho = 0.7;//挥发系数
	int left = 2;//信息素增量
	int Q = 400;//信息素增加数

	std::string filePath;

	std::cin >> filePath;

	std::ifstream in(filePath.c_str());
	if (!in.is_open()) {
		std::cout << "No exist!" << std::endl;
	}

	Point points[MAX_POINT_NUM];
	int index;
	double x, y;
	int cityNum = 0;
	while (in >> index >> x >> y) {
		points[cityNum++] = { index,x,y };
	}
	Point* devicePoints;
	float* deviceGraphMatrix;
	float* deviceHeuristicValue;
	float* deviceTau;
	float* deviceDeltaTau;
	CUDA_CALL(cudaMalloc(&devicePoints, sizeof(Point) * MAX_POINT_NUM));
	CUDA_CALL(cudaMemcpy(devicePoints, points, sizeof(Point) * MAX_POINT_NUM, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMalloc(&deviceGraphMatrix, sizeof(float) * MAX_POINT_NUM * MAX_POINT_NUM));
	CUDA_CALL(cudaMalloc(&deviceHeuristicValue, sizeof(float) * MAX_POINT_NUM * MAX_POINT_NUM));
	CUDA_CALL(cudaMalloc(&deviceTau, sizeof(float) * MAX_POINT_NUM * MAX_POINT_NUM));
	CUDA_CALL(cudaMalloc(&deviceDeltaTau, sizeof(float) * MAX_POINT_NUM * MAX_POINT_NUM));

	preProcessing <<<MAX_POINT_NUM, MAX_POINT_NUM >>>(devicePoints, deviceGraphMatrix, deviceHeuristicValue, deviceTau, deviceDeltaTau, cityNum);

	// init randomEngine
	curandState* devStates;
	CUDA_CALL(cudaMalloc(&devStates, antNum * sizeof(curandState)));
	setupCuRandKernel << <1, antNum >> > (devStates);

	float* deviceBestLength;
	int* deviceBestPath;
	cudaMalloc(&deviceBestPath, sizeof(int) * antNum * MAX_POINT_NUM);
	cudaMalloc(&deviceBestLength, sizeof(float) * antNum);
	initializeEachAntsBestLength<<<1, antNum>>>(deviceBestLength);

	int iterCounter = 0;

	cudaEvent_t startEvent;
	cudaEvent_t endEvent;
	float cudaElapsedTime = 0.0f;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&endEvent);
	cudaEventRecord(startEvent, 0);

	while (iterCounter < iterTimes) {
		constructPath<<<antNum, MAX_POINT_NUM>>>(deviceTau,deviceDeltaTau,deviceGraphMatrix,deviceHeuristicValue,devStates,cityNum,alpha,beta,Q,deviceBestLength,deviceBestPath);
		updatePheromones<<<MAX_POINT_NUM, MAX_POINT_NUM>>>(deviceTau,deviceDeltaTau,rho,cityNum);
		iterCounter++;
	}

	cudaEventRecord(endEvent, 0);
	cudaEventSynchronize(endEvent);
	cudaEventElapsedTime(&cudaElapsedTime, startEvent, endEvent);

	std::cout << "core kernel time:" << cudaElapsedTime << std::endl;

	float hostBestLength[antNum];
	int hostBestPath[antNum][MAX_POINT_NUM];

	cudaMemcpy(hostBestLength, deviceBestLength, sizeof(float) * antNum, cudaMemcpyDeviceToHost);
	cudaMemcpy(hostBestPath, deviceBestPath, sizeof(int) * antNum * MAX_POINT_NUM, cudaMemcpyDeviceToHost);

	float bestLength = hostBestLength[0];
	int pos = 0;
	for (int i = 1;i < antNum;i++) {
		if (hostBestLength[i] < bestLength) {
			bestLength = hostBestLength[i];
			pos = i;
		}
	}

	std::cout << bestLength << std::endl;
	for (int i = 0;i < cityNum;i++) {
		std::cout << hostBestPath[pos][i]+1 << ' ';
	}

	return 0;
}

//int main()
//{
//	std::string filePath;
//
//	std::cin >> filePath;
//
//	std::ifstream in(filePath.c_str());
//	if (!in.is_open()) {
//		std::cout << "No exist!" << std::endl;
//	}
//
//	std::vector<Point> points;
//	int index;
//	double x, y;
//	int cityNum = 0;
//	while (in >> index >> x >> y) {
//		points.push_back({ index,x,y });
//	}
//
//	double graphMatrix[MAX_POINT_NUM][MAX_POINT_NUM];
//	double heuristicValue[MAX_POINT_NUM][MAX_POINT_NUM];//启发式因子为两点间距离的倒数
//
//	for (auto &p : points) {
//		cityNum++;
//		for (auto &pr : points) {
//			double val = sqrt((p.x - pr.x)*(p.x - pr.x) + (p.y - pr.y)*(p.y - pr.y));
//			graphMatrix[p.index][pr.index] = graphMatrix[pr.index][p.index] = val;
//			heuristicValue[p.index][pr.index] = heuristicValue[pr.index][p.index] = 1 / val;
//		}
//	}
//
//	constexpr int antNum = 32;
//	constexpr int iterTimes = 800;
//	double alpha = 1;//信息素的重要程度
//	double beta = 3.8;//启发式因子的重要程度
//	double rho = 0.7;//挥发系数
//	int left = 2;//信息素增量
//	int Q = 400;//信息素增加数
//	double Tau[MAX_POINT_NUM][MAX_POINT_NUM];//全局的信息素信息
//
//
//	std::vector<int> bestPath(cityNum);
//	double bestLength = DBL_MAX;
//
//	for (int i = 1;i <= cityNum;i++) {
//		for (int j = 1;j <= cityNum;j++) {
//			Tau[i][j] = 0;
//		}
//	}
//
//	int iterCounter = 0;
//	while (iterCounter < iterTimes) {
//		std::default_random_engine randomEngine;
//		std::uniform_int_distribution<int> randomDistribution(1, cityNum);
//
//		double deltaTauTotal[MAX_POINT_NUM][MAX_POINT_NUM];
//		for (int i = 1;i <= cityNum;i++) {
//			for (int j = 1;j <= cityNum;j++) {
//				deltaTauTotal[i][j] = 0;
//			}
//		}
//
//		for (int i = 0;i < antNum;i++) {
//			//for each ant
//			bool allowed[MAX_POINT_NUM];
//			std::vector<int> path;
//			double currentDistance = 0;
//			clearAllowed(allowed, cityNum);
//
//			int startNum = randomDistribution(randomEngine);
//			path.push_back(startNum);
//			allowed[startNum] = false;
//
//			int current = startNum;
//			while (path.size() < cityNum) {
//				int next = calculateProbabilityAndSelect(Tau, heuristicValue, allowed, current, cityNum, alpha, beta);
//
//				//add it to path and update distance
//				path.push_back(next);
//				currentDistance += graphMatrix[current][next];
//
//				current = next;
//			}
//
//			calculateDeltaTauPerAnt(deltaTauTotal, Q, &path, currentDistance);
//
//			if (currentDistance < bestLength) {
//				bestLength = currentDistance;
//
//				std::copy(path.cbegin(), path.cend(), bestPath.begin());
//			}
//		}
//		updateTau(Tau, deltaTauTotal, rho, cityNum);
//
//		iterCounter++;
//	}
//	for (auto &p : bestPath) {
//		std::cout << p << ' ';
//	}
//	std::cout << '\n' << bestLength << std::endl;
//	getchar();
//	getchar();
//	return 0;
//}

