#ifdef OCTOTIGER_HAVE_CUDA

__global__ void kernel_reconstruct(double *Q, double *D1, double *U_, double *X, double omega) {
    bool first_thread = (blockIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0);
    if (first_thread)
        printf("Hello reconstruct");

}

#endif 