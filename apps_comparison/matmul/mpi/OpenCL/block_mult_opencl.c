
void multiply_block_accumulative (double *a, double *b, double *c, int workerRow, int workerColumn, int blockSieze, int matrixSize){
    cl_mem d_a, d_b, d_c;   // Matrices in device memory
    double start_time;      // Starting time
    double run_time;        // timing data
    char * kernelsource;    // kernel source string
    cl_int err;             // error code returned from OpenCL calls
    cl_device_id     device;        // compute device id
    cl_context       context;       // compute context
    cl_command_queue commands;      // compute command queue
    cl_program       program;       // compute program
    cl_kernel kernel; // compute kernel
    cl_uint deviceIndex = 0;
    cl_device_id devices[10];
    unsigned numDevices = getDeviceList(devices);
    device = devices[deviceIndex];   
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    commands = clCreateCommandQueue(context, device, 0, &err);
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, h_A, &err);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, h_B, &err);
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size, NULL, &err);
    kernelsource = getKernelSource("../C_elem.cl");
    program = clCreateProgramWithSource(context, 1, (const char **) & kernelsource, NULL, &err);
    free(kernelsource);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "mmul", &err);
    err =  clSetKernelArg(kernel, 0, sizeof(int),    &N);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_c);
    const size_t global[2] = {N, N};
    err = clEnqueueNDRangeKernel( commands, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
    err = clFinish(commands);
    err = clEnqueueReadBuffer( commands, d_c, CL_TRUE, 0, sizeof(float) * size, c, 0, NULL, NULL);
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    return 0;
}
char * getKernelSource(char *filename)
{
    FILE *file = fopen(filename, "r");
    if (!file){
        exit(1);
    }
    fseek(file, 0, SEEK_END);
    int len = ftell(file) + 1;
    rewind(file);
    char *source = (char *)calloc(sizeof(char), len);
    if (!source) {
        exit(1);
    }
    fread(source, sizeof(char), len, file);
    fclose(file);
    return source;
} 
