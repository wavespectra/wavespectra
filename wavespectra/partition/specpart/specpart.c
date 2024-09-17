#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "specpart.h"

static int nspec= 0, mk = -1, mth = -1, npart=0;
/*     ----------------------------------------------------------------
!       imi     i.a.   i   input discretized spectrum.
!       ind     i.a.   i   sorted addresses.
!       imo     i.a.   o   output partitioned spectrum.
!       zp      r.a.   i   spectral array.
!       npart   int.   o   number of partitions found.
!     ----------------------------------------------------------------*/
static int *neigh, *imi, *ind, *imo;

static float *zp;


void ptnghb();
void ptsort(int iihmax, int nnspec);


void pt_fld(int *imi,
            int *ind,
            int *imo,
            float *zp,
	    int ihmax);


void partinit(int nk,
              int nth) {
  
  if ( mk == nk && mth == nth)
    return;

  nspec = nk*nth;

  if (mk > 0) {
        free(neigh);
        free(imi);
        free(imo);
        free(ind);
        free(zp);
    }

    imi = (int *) malloc(nspec * sizeof(int));
    imo = (int *) malloc(nspec * sizeof(int));
    ind = (int *) malloc(nspec * sizeof(int));
    zp = (float *) malloc(nspec * sizeof(float));
    neigh = (int *) malloc(9 * nspec * sizeof(int));

    mk = nk;
    mth = nth;

    ptnghb();
}


void partition(float * spec,
	       int *ipart,
	       int nk,
	       int nth,
	       int ihmax) {

  double zmin, zmax, fact;
  int iang, ifreq, i;

  partinit(nk, nth);

  if ( nk != mk || nth != mth ) {
        printf("Error: partinit must be called with correct spectral dimensions\n");
        exit(EXIT_FAILURE);
    }

  // Copying content of spec into zp
  for ( iang = 0; iang < mth; iang++ ) { // Loop over points in the spectra
    for (ifreq = 0; ifreq < mk; ifreq++) { // Loop over spectra
      zp[ifreq + mk * iang] = spec[ifreq*mth + iang];
    }
  }

  zmin = zp[0];
  zmax = zp[0];

  for (i = 1; i < nspec; i++) {
    zmin = fmin(zmin, zp[i]);
    zmax = fmax(zmax, zp[i]);
  }

  // Mostly constant spectral array has no partitions
  if ( zmax - zmin < 1e-9 ) {
    for (i = 0; i < nspec; i++) {
      ipart[i] = 0;
    }
    npart = 0;
    return;
  }

  // shifting so that zmax becomes 0
  for (i = 0; i < nspec; i++) {
	zp[i] = zmax - zp[i];
  }

  // Bining in ihmax number of bins
  fact = (ihmax - 1.0) / (zmax - zmin);
  for (i = 0; i < nspec; i++) {
    imi[i] = fmax(0, fmin(ihmax-1, round(0.0 + zp[i] * fact)));  
  }

  // Fills the ind table with indexes that correspond to increasing levels of energy
  ptsort(ihmax, nspec);
  
  pt_fld(imi, ind, imo, zp, ihmax);
    
  for (iang = 0; iang < mth; iang++) {
	for (ifreq = 0; ifreq < mk; ifreq++) {
            ipart[ifreq + mk * iang] = imo[ifreq + mk * iang];
        }
    }  
}


void ptsort(int iihmax, int nnspec) {
    int i, in, iv;
    int * numv = malloc(iihmax*sizeof(int));
    int * iaddr = malloc(iihmax*sizeof(int));
    int * iorder = malloc(nnspec*sizeof(int));
    /*--------------------------------
     * 1.  counting occurences per height bins
     *-------------------------------*/
    for (i = 0; i < iihmax; i++) {
	numv[i] = 0;
    }

    for (i = 0; i < nspec; i++) {
      numv[imi[i]]++;
    }

    /*-----------------------------------
     *  2.  starting address per height (iaddr is cumulative distribution in bins)
     *----------------------------------*/
    iaddr[0] = 0;//1;
    for (i = 0; i < iihmax - 1; i++) {
        iaddr[i + 1] = iaddr[i] + numv[i];
    }

    /*-----------------------------------
     * 3.  order points
     *----------------------------------*/
    for (i = 0; i < nnspec; i++) {
      iv = imi[i];
      in = iaddr[iv];
      iorder[i] = in;
      iaddr[iv] = in + 1;
    }
    /*-----------------------------------                                                                                                                                                  
     * 4.  sort points                                                                                                                                                                    
     *----------------------------------*/
    for (i = 0; i < nnspec; i++) {
      ind[iorder[i]] = i;
    }
  free(numv);
  free(iaddr);
  free(iorder);
}

void ptnghb() {
  
  int n, j, i, k;
  
  neigh = (int *) malloc(9*nspec*sizeof(int));
  
  // ... base loop
  for (n = 0; n < nspec; n++) {
  // Replaced that one as easier to understand
  // All indices are shifted by 1 with respect to fortran
  // All intervals are of similar length
    j = n/mk;
    i = n - j*mk;
    k = -1;

    //  ... point at the left(0)
    if (i != 0) {
      k++;
      neigh[k + 9 * n] = n - 1;
    }

    // ... point at the right (1)
    if (i != mk - 1) {
      k++;
      neigh[k + 9 * n] = n + 1;
    }
    
    // ... point at the bottom(2)
    if (j != 0) {
      k++;
	neigh[k + 9 * n] = n - mk;
    }
      
    // ... add point at bottom_wrap to top
    if (j == 0) {
      k++;
      neigh[k + 9 * n] = nspec - (mk - i);
    }
    
    // ... point at the top(3)
    if (j != mth-1 ) {
      k++;
      neigh[k + 9 * n] = n + mk;
    }
    
      // ... add point to top_wrap to bottom
    if (j == mth-1) {
      k++;
      neigh[k + 9 * n] = n - (mth - 1) * mk;
    }
    
    // ... point at the bottom, left(4)
    if (i != 0 && j != 0) {
      k++;
      neigh[k + 9 * n] = n - mk - 1;
    }
    
    // ... point at the bottom, left with wrap.
    if (i != 0 && j == 0) {
      k++;
      neigh[k + 9 * n] = n - 1 + mk * (mth - 1);
    }
    
    // ... point at the bottom, right(5)
    if (i != mk - 1 && j != 0) {
      k++;
      neigh[k + 9 * n] = n - mk + 1;
    }
    
    // ... point at the bottom, right with wrap
    if (i != mk - 1 && j == 0) {
      k++;
      neigh[k + 9 * n] = n + 1 + mk * (mth - 1);
    }
    
    // ... point at the top, left(6)
    if (i != 0 && j != mth-1) {
      k++;
      neigh[k + 9 * n] = n + mk - 1;
    }
    
    // ... point at the top, left with wrap
    if (i != 0 && j == mth-1) {
      k++;
      neigh[k + 9 * n] = n - 1 - (mk) * (mth - 1);
    }
    
    // ... point at the top, right(7)
    if (i != mk-1 && j != mth-1) {
      k++;
      neigh[k + 9 * n] = n + mk + 1;
    }
      
    // ... point at top, right with wrap
    if (i != mk-1 && j == mth-1) {
      k++;
      neigh[k + 9 * n] = n + 1 - (mk) * (mth - 1);
    }
    
    neigh[8 + 9 * n] = k+1;
    // }
  }
}


int int_minval(int * data, int size) {
  int i, min;
  min = data[0];
  for ( i =1; i < size; i++ )
    min = fmin(data[i], min);
  return min;
}


int fifo_add(int * iq,
	     int iq_end,
	     int iv) {
  *(iq + iq_end) = iv;
  if ( iq_end > nspec-2 )
    return 0;
  return iq_end+1;
}

int fifo_empty(int iq_start,
	       int iq_end) {
  if (iq_start != iq_end)
    return 0;
  return 1;
}

int fifo_first(int * iq,
	       int * iq_start) {
  int iv = *(iq+*iq_start);
  *iq_start = *iq_start+1;
  if ( *iq_start > nspec -1 )
    *iq_start = 0;
  return iv;
} 

void pt_fld(int *imi,
	    int *ind,
	    int *imo,
	    float *zp,
	    int ihmax) {

  int mask, init, iwshed;
  int ic_label, ifict_pixel, m, ih, msave;
  int ip, i, ipp, ic_dist, iempty, ippp;
  int jl, jn, ipt, j;
  int iq_start, iq_end;
  float zpmax, ep1, diff;
  int * iq = malloc(nspec*sizeof(int));
  int * imd = malloc(nspec*sizeof(int));
  
  // 0.  initializations
  mask = -2;
  init = -1;
  iwshed = 0;
  for ( i = 0; i < nspec; i++ )
    imo[i] = init; // init; // Not sure what the point of using init is
  ic_label = 0;
  for ( i = 0; i < nspec; i++)
    imd[i] = 0;
  ifict_pixel = -100;

  // fifo inint to empty
  iq_start = 0;
  iq_end = 0;

  zpmax = zp[0];
  for ( i = 1; i < nspec; i++)
    zpmax = fmaxf(zpmax, zp[i]);

  // 1.  loop over levels (binned spectral values)
  m = 0;
  for ( ih = 0; ih < ihmax; ih++ ) {
    msave = m;

    // 1.a pixels at level ih
    // we pick bin ih and go over all points (from ind[msave] to when imi[ip] !ih
    for(;;) { // Infinite loop
      ip = ind[m];

      // Exit if all pixel at level ih have been looked at
      if ( imi[ip] != ih )
	break;

      // flag the point, if it stays flagge, it is a separate minimum.
      imo[ip] = mask;

      // consider neighbors. if there is neighbor, set distance and add
      // to queue.

      for ( i = 0; i < neigh[8+9*ip]; i++ ) {
	ipp = neigh[i+9*ip];
	if ( imo[ipp] > 0 || imo[ipp] == iwshed ) {
	  imd[ip] = 1;
	  iq_end = fifo_add (iq, iq_end, ip); // l305
	  break;
	}
      }

      // If we have been through all spectrum point then exit otherwise increment
      if ( m > nspec-2 )
	break;
      else
	m++;

    }


    //  1.b process the queue
    ic_dist = 1;
    // Mark end of fifo
    iq_end = fifo_add(iq, iq_end, ifict_pixel);
    for (;;) {
      ip = fifo_first(iq, &iq_start);
      // Check for end of processing
      if ( ip == ifict_pixel ) {
	if (fifo_empty(iq_start, iq_end))
	  break;
	else {
	  iq_end = fifo_add(iq, iq_end, ifict_pixel);
	  ic_dist++;
	  ip = fifo_first(iq, &iq_start);
	}
      }
      // Process queue
      for ( i = 0; i < neigh[8+9*ip]; i++ ) {
	ipp = neigh[i+9*ip];
	// Check for labeled watersheds or basins
	if ( (imd[ipp] < ic_dist) && ( (imo[ipp] > 0 ) || (imo[ipp] == iwshed))) {
	  if ( imo[ipp] > 0 ) {	    
	    if ( (imo[ip] == mask) || (imo[ip] == iwshed) )
	      imo[ip] = imo[ipp];
	    else if ( imo[ip] != imo[ipp] )
	      imo[ip] = iwshed;
	  }
	  else if (imo[ip] == mask)
	    imo[ip] = iwshed;
	}
	else if ( ( imo[ipp] == mask ) && (imd[ipp] == 0) ) {
	  imd[ipp] = ic_dist +1;
	  iq_end = fifo_add(iq, iq_end, ipp);
	}
	
      }
    }

    // 1.c Check for mask values in IMO to identify new basins
    m = msave;

    for (;;) {
      ip = ind[m];
      if ( imi[ip] != ih)
	break;
      imd[ip] = 0;

      if ( imo[ip] == mask ) {
	// ... New label for pixel
	ic_label++;
	iq_end = fifo_add(iq, iq_end, ip);
	imo[ip] = ic_label;
	// ... and all connected to it ...
	for (;;) {
	  if ( fifo_empty(iq_start, iq_end) )
	    break;
	  ipp = fifo_first(iq, &iq_start);

	  for ( i = 0; i < neigh[8+9*ipp]; i++ ) {
	    ippp = neigh[i+9*ipp];
	    if ( imo[ippp] == mask ) {
	      iq_end = fifo_add(iq, iq_end, ippp);
	      imo[ippp] = ic_label;
	    }
	  }
	
	}
      
      }

      if ( m > nspec-2 )
	break;
      else
	m++;
    }

  }
  
  
  /*  2.  Find nearest neighbor of 0 watershed points and replace
      use original input to check which group to affiliate with 0
      Soring changes first in IMD to assure symetry in adjustment.
  */
  for ( j = 0; j < 5; j++ ) {
    for ( i = 0; i < nspec; i++ )
      imd[i] = imo[i];
    for ( jl = 0; jl < nspec; jl++ ) {
      ipt = -1;
      if ( imo[jl] == 0 ) {
	ep1 = zpmax;
	for ( jn = 0; jn < neigh[8+9*jl]; jn++ ) {
	  diff = fabs(zp[jl]-zp[neigh[jn+9*jl]]);
	  if ( ( diff <= ep1 ) && ( imo[neigh[jn+9*jl]] != 0 )) {
	    ep1 = diff;
	    ipt = jn;
	  }
	}
	if ( ipt > -1 )
	  imd[jl] = imo[neigh[ipt+9*jl]];
      }
    }
    for ( i = 0; i < nspec; i++ )
      imo[i] = imd[i];
    if ( int_minval(imo, nspec) > 0 ){
      break;
    }
  }
 
  npart = ic_label;
  free(iq);
  free(imd);
}
