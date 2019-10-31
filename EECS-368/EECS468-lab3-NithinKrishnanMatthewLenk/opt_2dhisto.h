#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto(uint32_t*, size_t, size_t, uint8_t*, uint32_t* );
void* AllocateMemory(size_t);
void freeMemory(void*);
void ToDeviceMemory(void*, void*, size_t);
void ToHostMemory(void*, void*, size_t);

/* Include below the function headers of any other functions that you implement */

#endif
