################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/KltWithCov.cpp \
../src/img2pcd.cpp \
../src/imgFtRelated.cpp \
../src/input.cpp \
../src/kltUncertainty.cpp \
../src/main.cpp \
../src/pcVIsual.cpp \
../src/pclStereo.cpp \
../src/ppTransEst.cpp \
../src/stereoSolver.cpp \
../src/utility.cpp 

OBJS += \
./src/KltWithCov.o \
./src/img2pcd.o \
./src/imgFtRelated.o \
./src/input.o \
./src/kltUncertainty.o \
./src/main.o \
./src/pcVIsual.o \
./src/pclStereo.o \
./src/ppTransEst.o \
./src/stereoSolver.o \
./src/utility.o 

CPP_DEPS += \
./src/KltWithCov.d \
./src/img2pcd.d \
./src/imgFtRelated.d \
./src/input.d \
./src/kltUncertainty.d \
./src/main.d \
./src/pcVIsual.d \
./src/pclStereo.d \
./src/ppTransEst.d \
./src/stereoSolver.d \
./src/utility.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/local/include/vtk-5.10 -O0 -g3 -Wall -c -fmessage-length=0 -std=c++11 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


