#include <iostream>
#include <cmath>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <omp.h>  // GPU offload with OpenMP, parallel loops, and timing

using namespace std;

// ------------------------------------------------------------
// Global parameters
// ------------------------------------------------------------
const double gamma_val = 1.4; // Ratio of specific heats
const double CFL = 0.5;       // CFL number

// ------------------------------------------------------------
// Compute pressure from the conservative variables
// ------------------------------------------------------------
double pressure(double rho, double rhou, double rhov, double E) {
    double u = rhou / rho;
    double v = rhov / rho;
    double kinetic = 0.5 * rho * (u * u + v * v);
    return (gamma_val - 1.0) * (E - kinetic);
}

// ------------------------------------------------------------
// Compute flux in the x-direction
// ------------------------------------------------------------
void fluxX(double rho, double rhou, double rhov, double E,
           double& frho, double& frhou, double& frhov, double& fE) {
    double u = rhou / rho;
    double p = pressure(rho, rhou, rhov, E);
    frho = rhou;
    frhou = rhou * u + p;
    frhov = rhov * u;
    fE = (E + p) * u;
}

// ------------------------------------------------------------
// Compute flux in the y-direction
// ------------------------------------------------------------
void fluxY(double rho, double rhou, double rhov, double E,
           double& frho, double& frhou, double& frhov, double& fE) {
    double v = rhov / rho;
    double p = pressure(rho, rhou, rhov, E);
    frho = rhov;
    frhou = rhou * v;
    frhov = rhov * v + p;
    fE = (E + p) * v;
}

// ------------------------------------------------------------
// Simulation function with optional GPU offload
//Unifies simulation function: same function for both CPU and GPU, with GPU offload region inside
// ------------------------------------------------------------
void runSimulation(int Nx, int Ny, bool useGPU, double &time_out, vector<double>& kinetic_history) {
    const double Lx = 2.0;
    const double Ly = 1.0;
    double dx = Lx / Nx;
    double dy = Ly / Ny;
    int total_size = (Nx+2)*(Ny+2);

    // Allocate arrays
    double *rho = new double[total_size];
    double *rhou = new double[total_size];
    double *rhov = new double[total_size];
    double *E   = new double[total_size];
    double *rho_new = new double[total_size];
    double *rhou_new = new double[total_size];
    double *rhov_new = new double[total_size];
    double *E_new   = new double[total_size];
    bool *solid = new bool[total_size];

    // Cylinder obstacle
    const double cx = 0.5, cy = 0.5, radius = 0.1;

    // Free-stream
    const double rho0 = 1.0, u0 = 1.0, v0 = 0.0, p0 = 1.0;
    double E0 = p0/(gamma_val-1.0) + 0.5*rho0*(u0*u0 + v0*v0);

    // Initialize
    for(int i=0;i<Nx+2;i++){
        for(int j=0;j<Ny+2;j++){
            double x = (i-0.5)*dx;
            double y = (j-0.5)*dy;
            int idx = i*(Ny+2)+j;
            if ((x-cx)*(x-cx)+(y-cy)*(y-cy) <= radius*radius) {
                solid[idx]=true;
                rho[idx]=rho0; rhou[idx]=0.0; rhov[idx]=0.0; E[idx]=p0/(gamma_val-1.0);
            } else {
                solid[idx]=false;
                rho[idx]=rho0; rhou[idx]=rho0*u0; rhov[idx]=rho0*v0; E[idx]=E0;
            }
            rho_new[idx]=rhou_new[idx]=rhov_new[idx]=E_new[idx]=0.0;
        }
    }

    // CFL timestep
    double c0 = sqrt(gamma_val*p0/rho0);
    double dt = CFL * min(dx, dy) / ((fabs(u0)+fabs(v0)+c0));

    const int nSteps = 200; // for output demonstration
    // Start timing here, includes GPU offload region for fair comparison
    double start = omp_get_wtime();

    // >>> CHANGE: GPU offload region

    if(useGPU) {
        double *gpu_kinetic_dev = new double[nSteps]; // temporary on host
        //Copy data to GPU and keep it there for the entire simulation
        // Map all arrays to GPU, with tofrom for those that will be updated and read back
        // Avoids slow memory transfers each step by keeping data on GPU and only copying kinetic energy history back at the end
        #pragma omp target data map(tofrom: rho[0:total_size], rhou[0:total_size], rhov[0:total_size], E[0:total_size], \
                                    rho_new[0:total_size], rhou_new[0:total_size], rhov_new[0:total_size], E_new[0:total_size], \
                                    solid[0:total_size])
        {
            for(int n=0;n<nSteps;n++){
                // This moves loops to GPU and keeps data there, with only kinetic energy history copied back at the end
                // TEAMS FOR BLOCKS, PARALLEL FOR FOR CELLS, COLLAPSE(2) TO PARALLELIZE BOTH LOOPS
                #pragma omp target teams distribute parallel for collapse(2)
                for(int i=1;i<=Nx;i++){
                    for(int j=1;j<=Ny;j++){
                        int idx=i*(Ny+2)+j;
                        if(solid[idx]) { 
                            rho_new[idx]=rho[idx]; rhou_new[idx]=rhou[idx]; rhov_new[idx]=rhov[idx]; E_new[idx]=E[idx]; 
                            continue; 
                        }
                        // Lax-Friedrichs averaging
                        rho_new[idx] = 0.25*(rho[(i+1)*(Ny+2)+j] +rho[(i-1)*(Ny+2)+j]
                                     + rho[i*(Ny+2)+(j+1)] +rho[i*(Ny+2)+(j-1)]);
                        rhou_new[idx]=0.25*(rhou[(i+1)*(Ny+2)+j] + rhou[(i-1)*(Ny+2)+j]
                                     +rhou[i*(Ny+2)+(j+1)] + rhou[i*(Ny+2)+(j-1)]);
                        rhov_new[idx]=0.25*(rhov[(i+1)*(Ny+2)+j] + rhov[(i-1)*(Ny+2)+j]
                                     + rhov[i*(Ny+2)+(j+1)] + rhov[i*(Ny+2)+(j-1)]);
                        E_new[idx]   = 0.25*(E[(i+1)*(Ny+2)+j] + E[(i-1)*(Ny+2)+j]
                                     + E[i*(Ny+2)+(j+1)] + E[i*(Ny+2)+(j-1)]);
                        
                    }
                }
                // Copy updated values back
                #pragma omp target teams distribute parallel for collapse(2)
                for(int i=1;i<=Nx;i++){
                    for(int j=1;j<=Ny;j++){
                        int idx=i*(Ny+2)+j;
                        rho[idx]=rho_new[idx]; 
                        rhou[idx]=rhou_new[idx]; 
                        rhov[idx]=rhov_new[idx]; 
                        E[idx]=E_new[idx];
                    }
                }
                //Compute kinetic energy after updating all cells
                double total_kinetic_step = 0.0;
                //Compute kinetic energy in parallel on GPU, with reduction to sum across threads
                // Each thread contributes to total sum 
                #pragma omp target teams distribute parallel for collapse(2) \
                    reduction(+:total_kinetic_step)
                for(int i=1;i<=Nx;i++){
                    for(int j=1;j<=Ny;j++){
                        int idx=i*(Ny+2)+j;
                        double u = rhou[idx] / rho[idx];
                        double v = rhov[idx] / rho[idx];
                        total_kinetic_step += 0.5 * rho[idx] * (u*u + v*v);
                    }
                }
                // Save kinetics energy for this step
                gpu_kinetic_dev[n] = total_kinetic_step; // store on GPU and will be copied back
            }
        }
        // Copy kinetic energy history back to host
        kinetic_history.assign(gpu_kinetic_dev, gpu_kinetic_dev+nSteps);
        delete[] gpu_kinetic_dev;
    }
    else { // CPU version
        for(int n=0;n<nSteps;n++){
            double total_kinetic = 0.0; // will be accumulated on CPU
            for(int i=1;i<=Nx;i++){
                for(int j=1;j<=Ny;j++){
                    int idx=i*(Ny+2)+j;
                    if(solid[idx]) { rho_new[idx]=rho[idx]; rhou_new[idx]=rhou[idx]; rhov_new[idx]=rhov[idx]; E_new[idx]=E[idx]; continue; }
                    rho_new[idx] = 0.25*(rho[(i+1)*(Ny+2)+j]+rho[(i-1)*(Ny+2)+j]+rho[i*(Ny+2)+(j+1)]+rho[i*(Ny+2)+(j-1)]);
                    rhou_new[idx]=0.25*(rhou[(i+1)*(Ny+2)+j]+rhou[(i-1)*(Ny+2)+j]+rhou[i*(Ny+2)+(j+1)]+rhou[i*(Ny+2)+(j-1)]);
                    rhov_new[idx]=0.25*(rhov[(i+1)*(Ny+2)+j]+rhov[(i-1)*(Ny+2)+j]+rhov[i*(Ny+2)+(j+1)]+rhov[i*(Ny+2)+(j-1)]);
                    E_new[idx]   =0.25*(E[(i+1)*(Ny+2)+j]+E[(i-1)*(Ny+2)+j]+E[i*(Ny+2)+(j+1)]+E[i*(Ny+2)+(j-1)]);  
                }
            }
            for(int i=1;i<=Nx;i++)
                for(int j=1;j<=Ny;j++){
                    int idx=i*(Ny+2)+j;
                    rho[idx]=rho_new[idx]; rhou[idx]=rhou_new[idx]; rhov[idx]=rhov_new[idx]; E[idx]=E_new[idx];
                }
            
            //double total_kinetic = 0.0;
            for(int i=1;i<=Nx;i++)
                for(int j=1;j<=Ny;j++){
                    int idx=i*(Ny+2)+j;
                    double u=rhou[idx]/rho[idx], v=rhov[idx]/rho[idx];
                    total_kinetic += 0.5*rho[idx]*(u*u + v*v);
                }
            //to print table and make a comparison CPU vs GPU, we need to store kinetic energy history for CPU as well
            kinetic_history.push_back(total_kinetic); // store for CPU too
        }
    }
    //Used for speedup calculation, same timing for both CPU and GPU versions, with GPU offload region inside the simulation function
    time_out = omp_get_wtime()-start;

    delete[] rho; delete[] rhou; delete[] rhov; delete[] E;
    delete[] rho_new; delete[] rhou_new; delete[] rhov_new; delete[] E_new;
    delete[] solid;
}
// ------------------------------------------------------------
// Main function
// ------------------------------------------------------------
int main() {
    int Nx = 200, Ny = 100;
    vector<double> gpu_kinetic, cpu_kinetic;
    double cpu_time, gpu_time;

    // Run CPU
    runSimulation(Nx,Ny,false,cpu_time,cpu_kinetic);
    // Run GPU
    runSimulation(Nx,Ny,true,gpu_time,gpu_kinetic);

    // Print kinetic energy table
    cout << "\n===== KINETIC ENERGY TABLE =====\n";
    cout << "Step   CPU Kinetic    GPU Kinetic\n";
    for(size_t i=0;i<cpu_kinetic.size();i++){
        printf("%4zu %20.10f %20.10f\n", i, cpu_kinetic[i], gpu_kinetic[i]);
    }

    // Print performance
    cout << "\n===== PERFORMANCE =====\n";
    cout << "CPU time: " << cpu_time << " s\n";
    cout << "GPU time: " << gpu_time << " s\n";
    cout << "Speedup: " << cpu_time/gpu_time << "\n";

    cout << "\n===== PERFORMANCE TABLE =====\n";
    cout << " Nx      Ny      CPU(s)     GPU(s)     Speedup\n";
    // Test different grid sizes for performance comparison
    int sizes[4] = {1,4,8,16};

    for(int k=0;k<4;k++){
        int Nx_test = Nx * sizes[k];
        int Ny_test = Ny * sizes[k];

        vector<double> tmp1, tmp2;
        double cpu_t, gpu_t;

        runSimulation(Nx_test,Ny_test,false,cpu_t,tmp1);
        runSimulation(Nx_test,Ny_test,true,gpu_t,tmp2);

        printf("%6d %6d %10.4f %10.4f %10.2f\n",
            Nx_test, Ny_test, cpu_t, gpu_t, cpu_t/gpu_t);
    }
    return 0;
}