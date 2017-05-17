#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#if USE_MPI
#include <mpi.h>
#endif
#include "RngStream.h"
#include "bench_gtc.h"

real shift_t_comp=0.0;
real shift_t_comm1=0.0;
real shift_t_comm2=0.0;
real charge_t_comp=0.0;
real charge_t_comp_t1=0.0;
real charge_t_comm=0.0;
real charge_t_comm1=0.0;
real charge_t_comm2=0.0;
real charge_t_comm3=0.0;
real grid_t_comm1=0.0;
real grid_t_comm2=0.0;
real push_t_comp=0.0;

double timer() {

#if !USE_MPI
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + ((double) tp.tv_usec)*1e-6);
#else
    MPI_Barrier(MPI_COMM_WORLD);
    return MPI_Wtime();
#endif

}

int main (int argc, char** argv) {
    
    FILE* fp;
    gtc_bench_data_t* gtc_input;
    int micell;
    int mstep, ndiag, irun;
    
    int mype, numberpe, ntoroidal;
    
    double elt, elt_main, setup_time, push_time, shift_time, charge_time, 
      poisson_time,shift_toroidal_time, shift_radial_time, sorting_time,
      smooth_time, field_time, poisson_init_time, collision_time, remap_time, 
      moments_time, restart_time, total_time;

    setup_time = push_time = shift_time = charge_time = 
        poisson_time = poisson_init_time = smooth_time = sorting_time = collision_time = remap_time
        = field_time = total_time = shift_toroidal_time = shift_radial_time = moments_time = restart_time = 0.0;

#if USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numberpe);
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
#else
    numberpe = 1;
    mype = 0;
#endif
    
    if (mype == 0) {
    
        if (argc != 4) {
            usage(argv[0]);
#if USE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#else
            exit(1);
#endif
        }
    }

#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (mype == 0) {
        fp = fopen(argv[1], "r");
        if (fp == NULL) {
            fprintf(stderr, "*** Cannot open input file. ***\n");
            usage(argv[0]);
#if USE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#else
            exit(1);
#endif
        }
        fclose(fp);
    }

#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    micell = atoi(argv[2]);
    
    if (mype == 0) {
        if (micell <= 0) {
            fprintf(stderr, "*** Number of particles per cell should be a positive"
                           " integer. ***\n");
            usage(argv[0]);
 #if USE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#else
            exit(1);
#endif
        }
    }

#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    ntoroidal = atoi(argv[3]);
    
    if (mype == 0) {
        if (ntoroidal <= 0) {
            fprintf(stderr, "*** Number of toroidal domains should be a positive"
                           " integer. ***\n");
            usage(argv[0]);
 #if USE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#else
            exit(1);
#endif
        }
    }

#if USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    elt = timer();

    gtc_input = (gtc_bench_data_t *) malloc(sizeof(gtc_bench_data_t));

    (gtc_input->parallel_decomp).numberpe  = numberpe;
    (gtc_input->parallel_decomp).ntoroidal = ntoroidal;
    (gtc_input->parallel_decomp).mype = mype;
    
    read_input_file(argv[1], &(gtc_input->global_params),
		    &(gtc_input->parallel_decomp), &(gtc_input->radial_decomp));

    (gtc_input->global_params).micell = micell;

    setup(gtc_input);

    /* get particle data from input files */
    irun = (gtc_input->global_params).irun;
#if !GPU_ACCEL
    if (irun == 1) restart_read(gtc_input);
#endif

    chargei_init(gtc_input);

#if !GPU_ACCEL
    if (irun == 0) calc_moments(gtc_input);
#endif

#if GPU_ACCEL
    gpu_kernel_args_t *gpu_kernel_args = (gpu_kernel_args_t *) malloc(sizeof(gpu_kernel_args_t));
    gpu_setup(gtc_input, gpu_kernel_args);
    cpy_gtc_data_to_device(gtc_input, gpu_kernel_args);
    //#if FOURPOINT
    //    gpu_charge_init(gtc_input, gpu_kernel_args);
    //#endif
#endif

    setup_time = timer() - elt;
    
    mstep = (gtc_input->global_params).mstep;
    ndiag = (gtc_input->global_params).ndiag;
   
#ifdef __bgq__
    if (mype==0)
      print_mem_usage(&mype);
#endif

    elt_main = timer();

    for (istep=1; istep<=mstep; istep++) {

        for (irk=1; irk<=2; irk++) {

            /* idiag=0, do time history diagnosis */
            idiag = ((irk+1)%2) + istep%ndiag;
            //if (mype==0) printf("istep=%d irk=%d\n", istep, irk);
            /* smooth potentials, diagnostics */
            elt = timer();
	    //if (mype==0) printf("smooth\n");
            smooth(3, gtc_input);
            smooth_time += timer() - elt;

            /* field */
            elt = timer();
	    //if (mype==0) printf("field\n");
            field(gtc_input);
            field_time += timer() - elt;

            /* push ion */
            elt = timer();
	    //if (mype==0) printf("pushi\n");
#if GPU_ACCEL
	    gpu_pushi(gtc_input, gpu_kernel_args);
#else
            pushi(gtc_input);
#endif
            push_time += timer() - elt;

#if !GPU_ACCEL
            if (idiag == 0) 
	      diagnosis(gtc_input);
#endif

	    elt = timer();
#if USE_MPI

	    /* shift in toroidal and radial dimensions */
	    //if (mype==0) printf("shift_toroidal\n");
#if GPU_ACCEL
        #if SYNERGESTIC_SORT_SHIFT
	    gpu_bin_particles(gtc_input,gpu_kernel_args, 0);
	    gpu_shifti_toroidal(gtc_input,gpu_kernel_args);
        #else
	    gpu_shifti_toroidal(gtc_input,gpu_kernel_args);
            #if      PARTICLE_BINNING
	    gpu_bin_particles(gtc_input,gpu_kernel_args,0);
            #endif
        #endif
#else
            shifti_toroidal(gtc_input);
#endif
	    shift_toroidal_time += timer() - elt;

	    
	    elt = timer();
	    //if (mype==0) printf("shifti_radial\n"); 
	    if ((gtc_input->radial_decomp).nproc_radiald>1){
#if GPU_ACCEL
        #if SYNERGESTIC_SORT_SHIFT
	      gpu_bin_particles(gtc_input,gpu_kernel_args, 1);
	      gpu_shifti_radial(gtc_input,gpu_kernel_args);
        #else
	      gpu_shifti_radial(gtc_input,gpu_kernel_args);
              #if      PARTICLE_BINNING
	      gpu_bin_particles(gtc_input,gpu_kernel_args,1);
              #endif
        #endif
#else
	      shifti_radial(gtc_input);
#endif
            }
	    
	    shift_radial_time += timer() - elt;    
#endif
	    elt = timer();
	    //if (mype==0) printf("sorting\n");
#if !GPU_ACCEL
	    radial_bin_particles(gtc_input);
#endif
            sorting_time += timer() - elt;

	    //if (mype==0) printf("collision\n");
	    // Fokker Planck collision operator
#if !GPU_ACCEL
	    elt = timer();
	    //if (mype==0) printf("collision\n");
	    if (irk==2&&(gtc_input->global_params).tauii>0) 
	      collision(gtc_input);
	    collision_time += timer() - elt;

#if REMAPPING
            elt = timer();
            //if (mype==0) printf("remapping\n");
            if (irk==2&&istep%(gtc_input->particle_remap).remapping_freq==0)
              remap(gtc_input);
            remap_time += timer() - elt;
#endif
#endif

            /* ion perturbed density */
            elt = timer();
	    //if (mype==0) printf("charge\n");
#if GPU_ACCEL
	    gpu_chargei(gtc_input, gpu_kernel_args);
#else
            chargei(gtc_input);
#endif
            charge_time += timer() - elt;

            /* smooth ion density */
            elt = timer();
	    //if (mype==0) printf("smooth\n");
            smooth(0, gtc_input);
            smooth_time += timer() - elt;
	    
            /* solve GK Poission equation using adiabatic electron */
            elt = timer();
	    //if (mype==0) printf("poisson\n");
            poisson(0, gtc_input);
            if ((istep == 1) && (irk == 1))
                poisson_init_time = timer() - elt;
            else
                poisson_time += timer() - elt;
        }

	/* write particle data to restart files */
	elt = timer();
#if !GPU_ACCEL
	if (istep==mstep) 
	  restart_write(gtc_input);
#endif
	restart_time += timer() - elt;

	/* output moments for diagnosis */
	elt = timer();
#if !GPU_ACCEL
	if (((gtc_input->global_params).mstepall + istep)%(gtc_input->global_params).mmomentsoutput==0)
	  calc_moments(gtc_input);
#endif
	moments_time += timer() - elt;
    }
    total_time = timer() - elt_main;

    total_time = total_time - poisson_init_time;

    gtc_mem_free (gtc_input);
    free(gtc_input);
    
    real shift_t_comp_g=0.0;
    real shift_t_comm1_g=0.0;
    real shift_t_comm2_g=0.0;
    real charge_t_comp_g=0.0;
    real charge_t_comp_t1_g=0.0;
    real charge_t_comm_g=0.0;
    real charge_t_comm3_g=0.0;
    real grid_t_comm1_g=0.0;
    real grid_t_comm2_g=0.0;
    real push_t_comp_g=0.0;
    MPI_Reduce(&shift_t_comp, &shift_t_comp_g, 1, MPI_MYREAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&shift_t_comm1, &shift_t_comm1_g, 1, MPI_MYREAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&shift_t_comm2, &shift_t_comm2_g, 1, MPI_MYREAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&charge_t_comp, &charge_t_comp_g, 1, MPI_MYREAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&charge_t_comp_t1, &charge_t_comp_t1_g, 1, MPI_MYREAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&charge_t_comm, &charge_t_comm_g, 1, MPI_MYREAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&charge_t_comm3, &charge_t_comm3_g, 1, MPI_MYREAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&grid_t_comm1, &grid_t_comm1_g, 1, MPI_MYREAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&grid_t_comm2, &grid_t_comm2_g, 1, MPI_MYREAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&push_t_comp, &push_t_comp_g, 1, MPI_MYREAL, MPI_MAX, 0, MPI_COMM_WORLD);

    if (mype == 0) {
        fprintf(stderr, "%d time steps\n"
                "Total time:   %9.6f s\n"
                "Charge        %9.6f s (%3.4f)\n"
                "Push          %9.6f s (%3.4f)\n"
                "Shift_t       %9.6f s (%3.4f)\n"
                "Shift_r       %9.6f s (%3.4f)\n"
		"Sorting       %9.6f s (%3.4f)\n"
		"Collision     %9.6f s (%3.4f)\n"
		"Remapping     %9.6f s (%3.4f)\n"
                "Poisson       %9.6f s (%3.4f)\n"
                "Field         %9.6f s (%3.4f)\n"
                "Smooth        %9.6f s (%3.4f)\n"
		"Restart       %9.6f s (%3.4f)\n"
		"Moments       %9.6f s (%3.4f)\n"
                "Setup         %9.6f s\n"
                "Poisson Init  %9.6f s\n",
                mstep, total_time, 
                charge_time, (charge_time/total_time)*100.0,
                push_time, (push_time/total_time)*100.0,
                shift_toroidal_time, (shift_toroidal_time/total_time)*100.0,
		shift_radial_time, (shift_radial_time/total_time)*100.0,
                sorting_time, (sorting_time/total_time)*100.0,
		collision_time, (collision_time/total_time)*100.0,
		remap_time, (remap_time/total_time)*100.0,
                poisson_time, (poisson_time/total_time)*100.0,
                field_time, (field_time/total_time)*100.0,
                smooth_time, (smooth_time/total_time)*100.0,
		restart_time, (restart_time/total_time)*100.0,
		moments_time, (moments_time/total_time)*100.0,
                setup_time, poisson_init_time);

	fprintf(stderr, "shift_t_comp=%9.6f shift_t_comm1=%9.6f shift_t_comm2=%9.6f\n",
		shift_t_comp, shift_t_comm1, shift_t_comm2);
	fprintf(stderr, "shift_t_comp_g=%9.6f shift_t_comm1_g=%9.6f shift_t_comm2_g=%9.6f\n",
                shift_t_comp_g, shift_t_comm1_g, shift_t_comm2_g);
        fprintf(stderr, "charge_t_comp=%9.6f charge_t_comp_t1=%9.6f charge_t_comm=%9.6f charge_t_comm3=%9.6f\n",
                charge_t_comp, charge_t_comp_t1, charge_t_comm, charge_t_comm3);
        fprintf(stderr, "charge_t_comp_g=%9.6f charge_t_comp_t1_g=%9.6f charge_t_comm_g=%9.6f charge_t_comm3_g=%9.6f\n",
                charge_t_comp_g, charge_t_comp_t1_g, charge_t_comm_g, charge_t_comm3_g);
        fprintf(stderr, "grid_t_comm1=%9.6f grid_t_comm2=%9.6f\n",
                grid_t_comm1, grid_t_comm2);
        fprintf(stderr, "grid_t_comm1_g=%9.6f grid_t_comm2_g=%9.6f\n",
                grid_t_comm1_g, grid_t_comm2_g);
	fprintf(stderr, "push_t_comp=%9.6f\npush_t_comp_g=%9.6f\n",
		push_t_comp, push_t_comp_g);
    }
    
#if GPU_ACCEL
    if(mype == 0 || mype == numberpe - 1){
      printf(stderr, "at processor pe=%d\n", mype);
      print_gpu_timing(gpu_kernel_args);
    }
    free_gtc_data_on_device(gpu_kernel_args);
#endif

#if USE_MPI
    MPI_Finalize();
#endif

    return 0;
}

