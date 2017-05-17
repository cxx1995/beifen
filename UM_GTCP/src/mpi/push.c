#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include "bench_gtc.h"
#if USE_MPI
#include <mpi.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

int pushi(gtc_bench_data_t *gtc_input) {

    gtc_global_params_t     *params;
    gtc_field_data_t        *field_data;
    gtc_particle_data_t     *particle_data;
    gtc_aux_particle_data_t *aux_particle_data;
    gtc_radial_decomp_t     *radial_decomp;
    gtc_diagnosis_data_t    *diagnosis_data;
    gtc_particle_decomp_t   *parallel_decomp;
    
    int mpsi, mzeta, mflux; 
    int mi;
    int nbound;
    real delr, q0, q1, q2, rc, rw;
    real pi, pi2_inv, pi_inv;
    real a, a1, a0, qion, aion, gyroradius, nonlinear;
    real kappati, kappan, paranl;
    real delz, zetamin;

    real* __restrict__ z0;
    real* __restrict__ z1;
    real* __restrict__ z2;
    real* __restrict__ z3;
    real* __restrict__ z4;
    real* __restrict__ z5;
    real* __restrict__ z00;
    real* __restrict__ z01;
    real* __restrict__ z02;
    real* __restrict__ z03;
    real* __restrict__ z04;
    real* __restrict__ z05;
    real *__restrict__ z06;
    const real* __restrict__ wzion;
#if !ONTHEFLY_PUSHAUX
    const int* __restrict__ jtion0;
    const int* __restrict__ jtion1;
    const real* __restrict__ wtion0;
    const real* __restrict__ wtion1;
    const real* __restrict__ wpion;
#else
    real *delt; int *igrid, *mtheta;
    real *pgyro, *tgyro, *qtinv;
    real a_diff, smu_inv;
    int mpsi_max;
#endif
    int igrid_in, ipsi_in, ipsi_out, ipsi_valid_in, ipsi_valid_out;   

#if !ASSUME_MZETA_EQUALS1
    const int*  __restrict__ kzion;
#endif

    const real* __restrict__ evector;
    
    real *temp, *dtemp, *rtemi, *pfluxpsi, *rdtemi;
    real flow0, flow1, flow2;

    real sbound, rho_max;
    real psimax, psimin, cmratio, cinv, vthi, d_inv, ainv, tem_inv;
    //    real vdrtmp[5];
    real *vdrtmp;
    real dtime;
    int i;
    int ndiag;

    /*******/

    params            = &(gtc_input->global_params);
    field_data        = &(gtc_input->field_data);
    particle_data     = &(gtc_input->particle_data);
    aux_particle_data = &(gtc_input->aux_particle_data);
    radial_decomp     = &(gtc_input->radial_decomp);
    diagnosis_data    = &(gtc_input->diagnosis_data);
    parallel_decomp   = &(gtc_input->parallel_decomp);

    mzeta = params->mzeta; mpsi = params->mpsi;
    mi    = params->mi;
    pi = params->pi;
    nbound = params->nbound;
    a1 = params->a1; a0 = params->a0; a = params->a;
    qion = params->qion; aion = params->aion;
    gyroradius = params->gyroradius;
    mflux = params->mflux; nonlinear = params->nonlinear;
    delr = params->delr;
    q0 = params->q0; q1 = params->q1; q2 = params->q2;
    rc = params->rc; rw = params->rw;
    paranl = params->paranl; kappati = params->kappati; 
    kappan = params->kappan;
    flow0 = params->flow0;
    flow1 = params->flow1;
    flow2 = params->flow2;
    zetamin = params->zetamin;
    delz  = params->delz;
    ndiag = params->ndiag;

    z0 = particle_data->z0; z1 = particle_data->z1; 
    z2 = particle_data->z2; z3 = particle_data->z3;
    z4 = particle_data->z4; z5 = particle_data->z5;

    z00 = particle_data->z00; z01 = particle_data->z01; 
    z02 = particle_data->z02; z03 = particle_data->z03;
    z04 = particle_data->z04; z05 = particle_data->z05;
    // modified by bwang May 2013
#if TWO_WEIGHTS
    z06 = particle_data->z06;
#endif

#if !ONTHEFLY_PUSHAUX
    jtion0 = aux_particle_data->jtion0; 
    jtion1 = aux_particle_data->jtion1;
    wtion0 = aux_particle_data->wtion0;
    wtion1 = aux_particle_data->wtion1;
    wpion  = aux_particle_data->wpion;
    wzion  = aux_particle_data->wzion;
#endif

#if !ASSUME_MZETA_EQUALS1
    kzion  = aux_particle_data->kzion;
#endif

    evector = field_data->evector;
    temp = field_data->temp; dtemp = field_data->dtemp;
    rtemi = field_data->rtemi; pfluxpsi = field_data->pfluxpsi;
    vdrtmp = field_data->vdrtmp;

    igrid_in = radial_decomp->igrid_in;
    ipsi_in  = radial_decomp->ipsi_in;
    ipsi_out = radial_decomp->ipsi_out;
    ipsi_valid_in = radial_decomp->ipsi_valid_in;
    ipsi_valid_out = radial_decomp->ipsi_valid_out;
    rho_max = radial_decomp->rho_max;

    pi_inv = 1.0/pi;
    pi2_inv = 0.5*pi_inv;

#if ONTHEFLY_PUSHAUX
    delt = field_data->delt;
    igrid = field_data->igrid;
    mtheta = field_data->mtheta;
    pgyro = field_data->pgyro;
    tgyro = field_data->tgyro;
    qtinv = field_data->qtinv;
    mtheta = field_data->mtheta;
    a_diff = a1-a0;
    smu_inv = params->smu_inv;
    mpsi_max = mpsi-1;
#endif

   
    real *rmarker, *eflux, *hfluxpsi, *pmarki;
    real *hfluxpsi_all = (real *)_mm_malloc(((mpsi+1)+ 6*mflux)*sizeof(real), IDEAL_ALIGNMENT);
    real *dden = &hfluxpsi_all[mpsi+1];
    real *dden_all = &hfluxpsi_all[mpsi+1+mflux];
    real *dmark = &hfluxpsi_all[mpsi+1+2*mflux];
    real *dmark_all = &hfluxpsi_all[mpsi+1+3*mflux];
    real *dtem = &hfluxpsi_all[mpsi+1+4*mflux];
    real *dtem_all = &hfluxpsi_all[mpsi+1+5*mflux];

    assert(hfluxpsi_all!=NULL);

    rmarker = diagnosis_data->rmarker;
    eflux = diagnosis_data->eflux;
    rdtemi = diagnosis_data->rdtemi; 
    hfluxpsi = field_data->hfluxpsi;
    pmarki = field_data->pmarki;
    
    for (i=0; i<mflux; i++)
      {
        rmarker[i] = 0.0;
        eflux[i] = 0.0;
        dden[i] = 0.0;
        dmark[i] = 0.0;
        dden_all[i] = 0.0;
        dmark_all[i] = 0.0;
	dtem[i] = 0.0;
	dtem_all[i] = 0.0;
      }
#pragma omp parallel for
    for (i=0; i<mpsi+1; i++)
      {
        hfluxpsi[i] = 0.0;
        hfluxpsi_all[i] = 0.0;
      }
    real *scalar_data = diagnosis_data->scalar_data;

    /********/

    sbound = 1.0;
    if (nbound == 0)
        sbound = 0.0;

    psimax = 0.5 * a1 * a1;
    psimin = 0.5 * a0 * a0;

    /* paxis=0.5*(8.0*gyroradius)**2 */
    cmratio = qion/aion;
    cinv = 1.0/qion;
    vthi = gyroradius*fabs(qion)/aion;
    d_inv = ((real) mflux)/(a1-a0);
    tem_inv = 1.0/(aion*vthi*vthi);

    /* primary ion marker temperature and parallel flow velocity */
#pragma omp parallel for
    for (i=0; i<mpsi+1; i++) {
        temp[i]  = 1.0;
        dtemp[i] = 0.0;
        temp[i]  = 1.0/(temp[i] * rtemi[i] * aion * vthi * vthi); 
        /*inverse local temperature */
    }

    ainv  = 1.0/a;
    
    if (irk==1) {
        /* 1st step of Runge-Kutta method */
        dtime = 0.5*params->tstep;
        //vdrtmp[0] = 0.0; vdrtmp[1] = 0.0; vdrtmp[2] = 0.0;
        //vdrtmp[3] = 0.0; vdrtmp[4] = 0.0;
	for (i=0; i<mflux; i++){
	  vdrtmp[i] = 0.0;
	}

        //if (istep == 1) {
#pragma omp parallel for schedule(static)
            for (i=0; i<mi; i++) {
                if (z2[i] == HOLEVAL)
                    continue;
                z00[i] = z0[i];
                z01[i] = z1[i];
                z02[i] = z2[i];
                z03[i] = z3[i];
                z04[i] = z4[i];
		// modified by bwang May 2013
#if TWO_WEIGHTS
		z06[i] = z05[i];
#endif
            }
	    //} 
    } else {
     
        dtime = params->tstep;
        if (nonlinear < 0.5)  {
            fprintf(stderr, "Error! decoupling modes for "
                    "nonlinear = 0.0 not implemented\n");
            exit(1);
            //vdrtmp[0] = 0.0; vdrtmp[1] = 0.0; vdrtmp[2] = 0.0;
            //vdrtmp[3] = 0.0; vdrtmp[4] = 0.0;
	    for (i=0; i<mflux; i++){
	      vdrtmp[i] = 0.0;
	    }
        } else {
	  //vdrtmp[0] = pfluxpsi[0]; vdrtmp[1] = pfluxpsi[1]; 
	  //vdrtmp[2] = pfluxpsi[2]; vdrtmp[3] = pfluxpsi[3]; 
          //vdrtmp[4] = pfluxpsi[4];
	  for (i=0; i<mflux; i++){
	    vdrtmp[i] = pfluxpsi[i];
	  }
        }
    }

#if FINE_TIMER
    real start_t = MPI_Wtime();
#endif

    /*  update GC position  */
#pragma omp parallel 
{

    real r, rinv, tem, q, qinv, cost, sint, b, g, gp, ri, 
         rip, dbdp, dbdt, dedb, deni, upara, energy, rfac, 
         rfac1, rfac2, rfac3, kappa, 
         dptdp, dptdt, dptdz, 
         epara, vdr, wdrive, wpara, wdrift, wdot, pdot, tdot, zdot, rdot;
    int ii, ij1, ij2, ij3, ij4, idx1, idx2, idx3, idx4;
    int ip, i;
    real zion0m, zion1m, zion2m, zion3m, zion4m, zion5m;
    real zion00m, zion01m, zion02m, zion03m, zion04m, zion06m;
    real z1t, z2t;
    real e1, e2, e3, wz0, wz1, wt00, wt01, wt10, wt11,
         wp0, wp1;
    real wpi0, wpi1, wpi2;
    int kk, larmor;
    int m;
    real cdot;

#if ONTHEFLY_PUSHAUX
    real rhotmp, wzt, r_diff, rdum, tflr, tdumtmp, tdumtmp2, tdum, tdum2;
    real wtion0tmp, wtion1tmp;
    int iptmp, im, im2, idxpg, jt, jttmp, ipjt;
    int jtion0tmp, jtion1tmp, j00, j01;
#endif

    /*
    real efluxi, pfluxi, dflowi, entropyi;
    real sum_of_weights;
    real particles_energy[2];
    real vdrenergy;
    real *rmarker_temp = (real *)_mm_malloc((5*mflux+(mpsi+1))*sizeof(real), IDEAL_ALIGNMENT);
    real *eflux_temp = &rmarker_temp[mflux];
    real *dmark_temp = &rmarker_temp[2*mflux];
    real *dden_temp = &rmarker_temp[3*mflux];
    real *dtem_temp = &rmarker_temp[4*mflux];
    real *hfluxpsi_temp = &rmarker_temp[5*mflux];
     
    assert(rmarker_temp!=NULL);

    efluxi=0.0; pfluxi=0.0; dflowi=0.0; entropyi=0.0;
    particles_energy[0] = particles_energy[1] = 0.0;
    vdrenergy = 0.0;
    sum_of_weights = 0.0;
    for (i=0; i<mflux; i++)
      {
        rmarker_temp[i] = 0.0;
        eflux_temp[i] = 0.0;
        dmark_temp[i] = 0.0;
        dden_temp[i] = 0.0;
	dtem_temp[i] = 0.0;
      }
    for (i=0; i<mpsi+1; i++)
      {
        hfluxpsi_temp[i] = 0.0;
      }
    */

    //#pragma omp for schedule(static) nowait
#pragma omp for
    for (m=0; m<mi; m++) {

        zion2m = z2[m]; 
        
        /* skip holes */
        if (zion2m == HOLEVAL) {
            continue;
        } 

        zion0m = z0[m]; zion1m = z1[m]; 
        zion3m = z3[m];
        zion4m = z4[m]; zion5m = z5[m];

#if ASSUME_MZETA_EQUALS1
#if CPU_DEBUG
        assert(mzeta == 1);
#endif
        kk = 0;
        wz1 = (zion2m-zetamin)*delz;
#else
        kk = kzion[m];
        wz1 = wzion[m];
#endif

#if ONTHEFLY_PUSHAUX

        real psitmp   = zion0m;
        real thetatmp = zion1m; 
        real zetatmp  = zion2m;
        real rhoi     = zion5m*smu_inv;

#if SQRT_PRECOMPUTED
        r        = psitmp; 
#else
        r        = sqrt(2.0*psitmp);
#endif

        iptmp    = (int) ((r-a0)*delr+0.5);
        ip       = abs_min_int(mpsi, iptmp);

#if CPU_DEBUG
        assert(ip>=ipsi_valid_in);
        assert(ip<=ipsi_valid_out);
#endif

        jttmp    = (int) (thetatmp*pi2_inv*delt[ip]+0.5); 
        jt       = abs_min_int(mtheta[ip], jttmp);

        ipjt     = igrid[ip]+jt;
        
        wzt      = (zetatmp-zetamin)*delz;

        wz1 = wzt - (real) kk;
        wz0 = 1.0 - wz1;
        r_diff   = r-a0;

#else
        wz0 = 1.0-wz1;
#endif
	e1 = 0.0;
	e2 = 0.0;
	e3 = 0.0;

        for (larmor = 0; larmor < 4; larmor++) {

#if ONTHEFLY_PUSHAUX

            idxpg   = larmor + 4*(ipjt-igrid_in);
            //rdum    = delr * abs_min_real(a_diff,
            //          r_diff+rhoi*pgyro[idxpg]);

	    rhotmp = rhoi*pgyro[idxpg];
	    if (fabs(rhotmp)>rho_max) {
	      printf("warning: reducing rhoi to %e from %e\n", (rhotmp/fabs(rhotmp))*rho_max/pgyro[idxpg], rhoi);
	      rhotmp = (rhotmp/fabs(rhotmp))*rho_max;
	      rhoi = rhotmp/pgyro[idxpg];
	    }
	    rdum = delr * abs_min_real(a_diff,
				       r_diff+rhotmp);
            
            ii      = abs_min_int(mpsi_max, (int) rdum);

#if CPU_DEBUG
            assert(ii>=ipsi_in);
            assert(ii<=ipsi_out-1);
#endif

            wp1     = rdum - (real) ii;
            wp0     = 1.0  - wp1;

            /* Particle position in theta */
            tflr    = thetatmp + rhoi*tgyro[idxpg];
            /* Inner flux surface */
            /* Outer flux surface */
            
            im      = ii;
            im2     = ii + 1;
            
            tdumtmp = pi2_inv * (tflr - zetatmp * qtinv[im]) + 10.0;
            tdumtmp2 = pi2_inv * (tflr - zetatmp * qtinv[im2]) + 10.0;
            
            tdum = (tdumtmp - (int) tdumtmp) * delt[im];
            tdum2 = (tdumtmp2 - (int) tdumtmp2) * delt[im2];
            
            j00 = abs_min_int(mtheta[im]-1, (int) tdum);
            j01 = abs_min_int(mtheta[im2]-1, (int) tdum2);
            
            jtion0tmp = igrid[im] + j00;
            jtion1tmp = igrid[im2] + j01;
            
            wtion0tmp = tdum - (real) j00;
            wtion1tmp = tdum2 - (real) j01;

            ij1 = jtion0tmp - igrid_in;
            ij3 = jtion1tmp - igrid_in;
            wp0 = 1.0 - wp1;
            wt10 = wtion0tmp;
            wt11 = wtion1tmp;
            wt01 = 1.0 - wt11;
            wt00 = 1.0 - wt10;
            ij2  = ij1 + 1;
            ij4  = ij3 + 1;

#else
            ij1 = jtion0[4*m+larmor] - igrid_in;
            ij3 = jtion1[4*m+larmor] - igrid_in;
            wp0 = 1.0 - wpion[4*m+larmor];
            wt00 = 1.0 - wtion0[4*m+larmor];
            wt01 = 1.0 - wtion1[4*m+larmor];
            ij2 = ij1 + 1;
            ij4 = ij3 + 1;
            wp1  = 1.0 - wp0;
            wt10 = 1.0 - wt00;
            wt11 = 1.0 - wt01;

#endif

#if !ASSUME_MZETA_EQUALS1
            idx1 = 3*(mzeta+1)*ij1+3*kk;
            idx2 = 3*(mzeta+1)*ij2+3*kk;
            idx3 = 3*(mzeta+1)*ij3+3*kk;
            idx4 = 3*(mzeta+1)*ij4+3*kk;


            e1 = e1 + wp0 * wt00 * (wz0 * evector[idx1+0]  
                    + wz1 * evector[idx1+3]);
            e2 = e2 + wp0 * wt00 * (wz0 * evector[idx1+1] 
                    + wz1 * evector[idx1+4]);
            e3 = e3 + wp0 * wt00 * (wz0 * evector[idx1+2] 
                    + wz1 * evector[idx1+5]);
        
            e1 = e1 + wp0 * wt10 * (wz0 * evector[idx2+0]   
                    + wz1 * evector[idx2+3]);
            e2 = e2 + wp0 * wt10 * (wz0 * evector[idx2+1] 
                    + wz1 * evector[idx2+4]);
            e3 = e3 + wp0 * wt10 * (wz0 * evector[idx2+2] 
                    + wz1 * evector[idx2+5]);
        
            e1 = e1 + wp1 * wt01 * (wz0 * evector[idx3+0]   
                    + wz1 * evector[idx3+3]);
            e2 = e2 + wp1 * wt01 * (wz0 * evector[idx3+1] 
                    + wz1 * evector[idx3+4]);
            e3 = e3 + wp1 * wt01 * (wz0 * evector[idx3+2] 
                    + wz1 * evector[idx3+5]);
        
            e1 = e1 + wp1 * wt11 * (wz0 * evector[idx4+0]   
                    + wz1 * evector[idx4+3]);
            e2 = e2 + wp1 * wt11 * (wz0 * evector[idx4+1] 
                    + wz1 * evector[idx4+4]);
            e3 = e3 + wp1 * wt11 * (wz0 * evector[idx4+2] 
                    + wz1 * evector[idx4+5]);
#else

            real wpt000 = wp0 * wt00;
            real wpt010 = wp0 * wt10;
            real wpt101 = wp1 * wt01;
            real wpt111 = wp1 * wt11;

            idx1 = 6*ij1;
            idx3 = 6*ij3;

            real evector10 = evector[idx1+0];
            real evector11 = evector[idx1+1];
            real evector12 = evector[idx1+2];
            real evector13 = evector[idx1+3];
            real evector14 = evector[idx1+4];
            real evector15 = evector[idx1+5];
            real evector20 = evector[idx1+6];
            real evector21 = evector[idx1+7];
            real evector22 = evector[idx1+8];
            real evector23 = evector[idx1+9];
            real evector24 = evector[idx1+10];
            real evector25 = evector[idx1+11];

            real evector30 = evector[idx3+0];
            real evector31 = evector[idx3+1];
            real evector32 = evector[idx3+2];
            real evector33 = evector[idx3+3];
            real evector34 = evector[idx3+4];
            real evector35 = evector[idx3+5];
            real evector40 = evector[idx3+6];
            real evector41 = evector[idx3+7];
            real evector42 = evector[idx3+8];
            real evector43 = evector[idx3+9];
            real evector44 = evector[idx3+10];
            real evector45 = evector[idx3+11];

            e1 = e1 + wpt000 * (wz0 * evector10 + wz1 * evector13)
                    + wpt010 * (wz0 * evector20 + wz1 * evector23)
                    + wpt101 * (wz0 * evector30 + wz1 * evector33)
                    + wpt111 * (wz0 * evector40 + wz1 * evector43);
        
            e2 = e2 + wpt000 * (wz0 * evector11 + wz1 * evector14)
                    + wpt010 * (wz0 * evector21 + wz1 * evector24)
                    + wpt101 * (wz0 * evector31 + wz1 * evector34)
                    + wpt111 * (wz0 * evector41 + wz1 * evector44);
            
            e3 = e3 + wpt000 * (wz0 * evector12 + wz1 * evector15)
                    + wpt010 * (wz0 * evector22 + wz1 * evector25)
                    + wpt101 * (wz0 * evector32 + wz1 * evector35)
                    + wpt111 * (wz0 * evector42 + wz1 * evector45);
#endif

        } 

        wpi0 = 0.25 * e1;
        wpi1 = 0.25 * e2;
        wpi2 = 0.25 * e3;

        if (irk == 1) {
            zion00m = zion0m;
            zion01m = zion1m;
            zion02m = zion2m;
            zion03m = zion3m;
            zion04m = zion4m;
	    // modified by bwang May 2013
#if TWO_WEIGHTS
	    zion06m = z06[m];
#endif
        } else {
       
            zion00m = z00[m]; zion01m = z01[m]; zion02m = z02[m]; 
            zion03m = z03[m]; zion04m = z04[m];
	    // modified by bwang May 2013 
#if TWO_WEIGHTS
	    zion06m = z06[m];
#endif
        }
       
#if !ONTHEFLY_PUSHAUX
#if SQRT_PRECOMPUTED
	r = zion0m;
#else
	r = sqrt(2.0 * zion0m);
#endif
#endif
        rinv = 1.0/r;
        
        ii = abs_min_int(mpsi-1, ((int)((r-a0)*delr)));
        ip = abs_min_int(mflux-1, ((int)((r-a0)*d_inv)));

        wp0 = ((real)(ii+1)) - (r-a0)*delr;
        wp1 = 1.0 - wp0;
        
        tem = wp0 * temp[ii] + wp1 * temp[ii+1];
        q = q0 + q1 * r * ainv + q2 * r * r * ainv * ainv;
        qinv = 1.0/q;
        
        cost = cos(zion1m);
        sint = sin(zion1m);

        /* cost0=cos(zion(2,m)+r*sint) */
        /* sint0=sin(zion(2,m)+r*sint) */
        b = 1.0/(1.0 + r * cost);
        g = 1.0;
        gp = 0.0;

        /* ri=r*r*qinv */
        /* rip=(2.0*q0+q1*r*ainv)*qinv*qinv */
        ri = 0.0;
        rip = 0.0;
        dbdp = -1.0 * b * b * cost * rinv;
        dbdt = b * b * r * sint;
        dedb = cinv * (zion3m * zion3m * qion * b * cmratio 
                + zion5m * zion5m);
     
        deni = 1.0/(g*q + ri + zion3m*(g*rip-ri*gp));
        upara = zion3m * b * cmratio;
        energy = 0.5 * aion * upara * upara + zion5m * zion5m * b;
        
        rfac1 = rw * (r-rc);
#if PROFILE_SHAPE==0
	// exp(x^6)
        rfac2 = rfac1 * rfac1;
        rfac3 = rfac2 * rfac2 * rfac2;
        rfac = exp(-1.0*rfac3);
#elif PROFILE_SHAPE==1
	// sech2(x)
	rfac2 = tanh(rfac1)*tanh(rfac1);
	rfac3 = 1.0 - rfac2;
	rfac = rfac3;
#endif	
        kappa = 1.0 - sbound + sbound * rfac;
        
        /* kappa=((gtc_min(umax*umax,energy*tem)-1.5)*kappati+kappan)*kappa*rinv */
        kappa = ((energy * tem - 1.5) * kappati + kappan) * kappa * rinv;

        /* perturbed quantities */
        dptdp = wpi0;
        dptdt = wpi1;
        dptdz = wpi2 - wpi1 * qinv;
        epara =-1.0 * wpi2 * b * q * deni;

        /* subtract net particle flow */
        dptdt = dptdt + vdrtmp[ip];

        /* ExB drift in radial direction for w-dot and flux diagnostics */
        vdr = q * (ri * dptdz - g * dptdt) * deni;
        wdrive = vdr * kappa;
        wpara = epara * (upara-dtemp[ii]) * qion * tem;
        wdrift = q * (g * dbdt * dptdp - g * dbdp * dptdt + ri * dbdp * dptdz) 
                   * deni * dedb * qion * tem;
	//paranl should not be in wdot, bwang 2013
        //wdot = (z05[m] - paranl * zion4m) * (wdrive + wpara + wdrift);
	//wdot = (z05[m] - zion4m)*(wdrive + wpara + wdrift);
	//wdot = wdrive + wpara + wdrift;
	// modified by bwang May 2013
#if TWO_WEIGHTS
	wdot = z05[m] * (wdrive + wpara + wdrift);
	cdot = -((gp*zion3m - 1.0)*paranl*dptdt - paranl*q*(1.0+rip*zion3m)*dptdz)*deni*(upara-dtemp[ii]) * aion * tem * z05[m];
#else
	wdot = (z05[m] - paranl * zion4m) * (wdrive + wpara + wdrift);
#endif

        /* self-consistent and external electric field for marker orbits */
        dptdp = dptdp*nonlinear+gyroradius*(flow0+flow1*r*ainv+flow2*r*r*ainv*ainv);
        dptdt = dptdt*nonlinear;
        dptdz = dptdz*nonlinear;

        /* particle velocity */
        pdot = q*(-g*dedb*dbdt - g*dptdt + ri*dptdz)*deni;
        tdot = (upara*b*(1.0-q*gp*zion3m) + q*g*(dedb*dbdp + dptdp))*deni;
        zdot = (upara*b*q*(1.0+rip*zion3m) - q*ri*(dedb*dbdp + dptdp))*deni;
        rdot = ((gp*zion3m-1.0)*(dedb*dbdt + paranl*dptdt)-paranl*q*(1.0+rip*zion3m)*dptdz)*deni;
         
        /* update particle position */
#if SQRT_PRECOMPUTED
	zion0m = gtc_max(1.0e-8 * psimax, 0.5*zion00m*zion00m + dtime*pdot);
	z0[m] = sqrt(2.0*zion0m);
#else
        z0[m]  = gtc_max(1.0e-8 * psimax, zion00m + dtime*pdot);
#endif
        zion1m = zion01m + dtime*tdot;
        zion2m = zion02m + dtime*zdot;
        z3[m]  = zion03m + dtime*rdot;
        z4[m]  = zion04m + dtime*wdot;
	// modified by bwang May 2013
#if TWO_WEIGHTS
	z05[m] = zion06m + dtime*cdot;
#endif
        /*  theta and zeta normalize to [0,2*pi), modulo is slower than hand coded
            procedure on Seaborg. However, modulo works better and is preferable.
            zion(2,m)=zion(2,m)*pi2_inv+10.0 !period of 1
            zion(2,m)=2.0*pi*(zion(2,m)-aint(zion(2,m))) ![0,2*pi)
            zion(3,m)=zion(3,m)*pi2_inv+10.0
            zion(3,m)=2.0*pi*(zion(3,m)-aint(zion(3,m))) */

        z1t   = zion1m*pi2_inv+10.0;
        z1[m] = 2.0*pi*(z1t - ((int)z1t));

        z2t   = zion2m*pi2_inv+10.0;
        z2[m] = 2.0*pi*(z2t - ((int)z2t));

        /*  UPDATE: 02/10/2006  The modulo function now streams on the X1E
            02/20/2004  The modulo function seems to prevent streaming on the X1
            02/20/2004 mod() does the same thing as modulo but it streams.
            02/23/2004 need to do mod((pi2+zion),pi2) instead of mod(zion,pi2) in order
            to catch the negative values of zion.
            zion(2,m)=mod((pi2+zion(2,m)),pi2)
            zion(3,m)=mod((pi2+zion(3,m)),pi2) */
        
        if (irk == 2) {
#if SQRT_PRECOMPUTED
	  if (z0[m] > a1) {
#else
            if (z0[m] > psimax) {
#endif
                z0[m] = zion00m;
                z1[m] = 2.0 * pi - zion01m;
                z2[m] = zion02m;
                z3[m] = zion03m;
                z4[m] = zion04m;
		// modified by bwang May 2013
#if TWO_WEIGHTS
		z05[m] = zion06m;
#endif

#if SQRT_PRECOMPUTED
	    } else if (z0[m] < a0) {
#else
            } else if (z0[m] < psimin) {
#endif
                z0[m] = zion00m;
                z1[m] = 2.0 * pi - zion01m;
                z2[m] = zion02m;
                z3[m] = zion03m;
                z4[m] = zion04m;
		// modified by bwang May 2013
#if TWO_WEIGHTS
		z05[m] = zion06m;
#endif
            }

            /* writing data to z00 here instead of the next
             * iteration (irk = 1) */
	    // move to irk = 1 of next iteration
	    /*
            z00[m] = z0[m];
            z01[m] = z1[m];
            z02[m] = z2[m];
            z03[m] = z3[m];
            z04[m] = z4[m];
	    // modified by bwang May 2013
#if TWO_WEIGHTS
	    z06[m] = z05[m];
#endif
	    */
	  }

        /*************** for diagnosis ***********/  
	  /*    	
	if (idiag==0) {
	  // irk==1 diagnosis
	  ip = abs_min_int(mflux-1, (int)((r-a0)*d_inv));
	  ii = abs_min_int(mpsi, (int)((r-a0)*delr+0.5));
          vdrenergy = vdr*rinv*(energy-1.5*aion*vthi*vthi*rtemi[ii])*zion04m;
          // radial profile of heat flux
          hfluxpsi_temp[ii] = hfluxpsi_temp[ii] + vdrenergy;

          // marker, energy, particle, momentum fluxes, parallel flow, entropy and kinetic energy
          rmarker_temp[ip] = rmarker_temp[ip] + 1.0;
          eflux_temp[ip] = eflux_temp[ip] + vdrenergy;

          efluxi = efluxi + vdrenergy;
          pfluxi = pfluxi + vdr*rinv*zion04m;
          dflowi = dflowi + b*zion03m*zion04m;
          entropyi = entropyi + zion04m*zion04m;

          particles_energy[0] += energy*zion04m;
          particles_energy[1] += energy;

          dmark_temp[ip] = dmark_temp[ip] + vdr*rinv*r;
	  dden_temp[ip] = dden_temp[ip] + 1.0;
	  sum_of_weights = sum_of_weights + zion04m;
	}

	if (nonlinear>0.5&&paranl<0.5) {
	  //if (nonlinear>0.5) {
	  if (istep%ndiag==0) {
	    if (irk==2){
	      zion0m = z0[m]; zion1m = z1[m];
	      zion3m = z3[m]; zion4m = z4[m];
              zion5m = z5[m];
#if SQRT_PRECOMPUTED 
	      r = zion0m;
#else
	      r = sqrt(2.0*zion0m);
#endif
	      ip = abs_min_int(mflux-1, (int)((r-a0)*d_inv));
	      cost = cos(zion1m);
	      b = 1.0/(1.0 + r * cost);
	      upara = zion3m * b * cmratio;
	      energy = 0.5 * aion * upara * upara + zion5m * zion5m * b;

	      dden_temp[ip] = dden_temp[ip] + 1.0;
	      dtem_temp[ip] = dtem_temp[ip] + energy*zion4m;
	    }
	  }
	}
	  */
	} // end m

	/*
#pragma omp critical 
 {
   if (idiag==0){
     for (i=0; i<mflux; i++) {
       rmarker[i] = rmarker[i] + rmarker_temp[i];
       eflux[i] = eflux[i] + eflux_temp[i];
       dmark[i] = dmark[i] + dmark_temp[i];
       dden[i] = dden[i] + dden_temp[i];
     }
     for (i=0; i<mpsi+1; i++){
       hfluxpsi[i] = hfluxpsi[i] + hfluxpsi_temp[i];
     }
     scalar_data[0] += efluxi;
     scalar_data[2] += pfluxi;
     scalar_data[6] += dflowi;
     scalar_data[8] += entropyi;
     scalar_data[12] += particles_energy[0];
     scalar_data[13] += particles_energy[1];

     scalar_data[15] += sum_of_weights;
   }

   if (nonlinear>0.5&&paranl<0.5) {
     //if (nonlinear>0.5){
     if (istep%ndiag==0) {      
       if (irk==2){
	 for (i=0; i<mflux; i++){
	   dden[i] = dden[i] + dden_temp[i];
	   dtem[i] = dtem[i] + dtem_temp[i];
	 }
       }
     }
   }
 }

 _mm_free(rmarker_temp);
	*/
    }

#if FINE_TIMER
    real end_t = MPI_Wtime();
    push_t_comp += (end_t - start_t);
#endif

    /*
#if USE_MPI
 real tdum = 0.01*ndiag;
 if (idiag==0){
   MPI_Allreduce(hfluxpsi, hfluxpsi_all, mpsi+1, MPI_MYREAL, MPI_SUM, MPI_COMM_WORLD);
   for (int i=0; i<mpsi+1; i++){
     hfluxpsi[i] = hfluxpsi_all[i]*pmarki[i];
   }
   
   MPI_Allreduce(dmark, dmark_all, mflux, MPI_MYREAL, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(dden, dden_all, mflux, MPI_MYREAL, MPI_SUM, MPI_COMM_WORLD);
   for (int i=0; i<mflux; i++){
     dmark[i] = dmark_all[i]/gtc_max(1.0, dden_all[i]);
     pfluxpsi[i] = (1.0-tdum)*pfluxpsi[i] + tdum*dmark[i];
   }
 }

 if (nonlinear>0.5&&paranl<0.5) {
   //if (nonlinear>0.5) {
   if (istep%ndiag==0) {
     if (irk==2){
       MPI_Allreduce(dden, dden_all, mflux, MPI_MYREAL, MPI_SUM, MPI_COMM_WORLD);
       MPI_Allreduce(dtem, dtem_all, mflux, MPI_MYREAL, MPI_SUM, MPI_COMM_WORLD);
       for (int i=0; i<mflux; i++){
	 dtem[i] = dtem_all[i]*tem_inv/gtc_max(1.0, dden_all[i]);
	 rdtemi[i] = (1.0-tdum)*rdtemi[i] + tdum*dtem[i];
       }
     }
   }
 }
#else
 real tdum=0.01*ndiag;
 if (idiag==0){
   for (int i=0; i<mpsi+1; i++){
     hfluxpsi[i] = hfluxpsi[i]*pmarki[i];
   }
   
   for (int i=0; i<mflux; i++){
     dmark[i] = dmark[i]/gtc_max(1.0, dden[i]);   
     pfluxpsi[i] = (1.0-tdum)*pfluxpsi[i] + tdum*dmark[i];
   }
 }
 
 if (irk==2){
   if (nonlinear>0.5&&paranl<0.5) {
     //if (nonlinear>0.5) {
     if (istep%ndiag==0) {
       for (int i=0; i<mflux; i++){
	 dtem[i] = dtem[i]*tem_inv/gtc_max(1.0, dden[i]);
	 rdtemi[i] = (1.0-tdum)*rdtemi[i] + tdum*dtem[i];   
       }
     }
   }
 }
#endif
 
   // restore temprature profile when running a nonlinear calculation (nonlinear=1.0)
   // without the velocity space nonlinearity (paranl=0.0)
 if (nonlinear>0.5&&paranl<0.5) {
   //if (nonlinear>0.5){
   if (istep%ndiag==0) {
     if (irk==2){
#pragma omp parallel 
{
  int m, ip;
  real zion0m, zion1m, zion2m, zion3m, zion4m, zion5m;
  real r, cost, b, upara, energy;
#pragma omp for schedule(static) nowait
  for (m=0; m<mi; m++){
    zion2m = z2[m];
    // skip holes
    if (zion2m == HOLEVAL) {
      continue;
    }
    zion0m = z0[m]; zion1m = z1[m];
    zion3m = z3[m]; zion4m = z4[m]; 
    zion5m = z5[m];
    
#if SQRT_PRECOMPUTED
    r        = zion0m;
#else
    r        = sqrt(2.0*zion0m);
#endif
    
    ip = abs_min_int(mflux-1, (int)((r-a0)*d_inv));
    
    cost = cos(zion1m);
    b = 1.0/(1.0 + r * cost);
    upara = zion3m * b * cmratio;
    energy = 0.5 * aion * upara * upara + zion5m * zion5m * b;    
    //Z Lin's heat bath version
    //zion4m = zion4m - (energy*tem_inv - 1.5)*rdtemi[ip];
    
    // corrction to include paranl effect according to weixing
    // the correction may not be necessary according to Jerome
    // because the restoration uses delta_f/f_0
    //zion4m = zion4m - (1.0-paranl*zion4m)*(energy*tem_inv - 1.5)*rdtemi[ip];
    //remove paranl above since parnal should not be in wdot bwang 2013
    zion4m = zion4m - (1.0-zion4m)*(energy*tem_inv - 1.5)*rdtemi[ip];
    z4[m] = zion4m;
    z04[m] = zion4m;
  }
 }
     }  
   }
 }
    */

 _mm_free(hfluxpsi_all);
 
    return 0;
}
