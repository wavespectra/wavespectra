! -*- f90 -*-
      module specpart

      private

      integer :: ihmax=200, nspec= 0, mk = -1, mth = -1, npart=0
!     ----------------------------------------------------------------
!       imi     i.a.   i   input discretized spectrum.
!       ind     i.a.   i   sorted addresses.
!       imo     i.a.   o   output partitioned spectrum.
!       zp      r.a.   i   spectral array.
!       npart   int.   o   number of partitions found.
!     ----------------------------------------------------------------
      integer, allocatable :: neigh(:,:),imi(:),ind(:),imo(:)
      real, allocatable :: zp(:)
      
      public :: partinit, partition, ihmax, npart

      contains
      
      subroutine partinit(nk,nth)
      
      if ( mk.eq.nk .and. mth.eq.nth ) return
      nspec=nk*nth
      
      if ( mk.gt.0 ) deallocate ( neigh, imi, imo, ind, zp )
      allocate(imi(nspec), imo(nspec), ind(nspec), zp(nspec))
      allocate ( neigh(9,nspec) )
      mk     = nk
      mth    = nth
      call ptnghb
      
      end subroutine partinit
      
      
      subroutine partition(spec,ipart,nk,nth)
      
      real, intent(in) :: spec(nk,nth)
      integer, intent(out) :: ipart(nk,nth)  
      integer iang,nk,nth
      real zmin,zmax
      
      call partinit(nk,nth)
      
      if (nk.ne.mk.or.mth.ne.nth) then
        write(*,*) mk,mth,nk,nth
        write(*,*) 'Error: partinit must be called with correct spectral dimensions'
        stop
      endif
      
      do iang=1, mth
        zp(1+(iang-1)*mk:iang*mk) = spec(:,iang)
      enddo
      zmin=minval(zp)
      zmax=maxval(zp)
      if (zmax-zmin.lt.1.e-9) then
        ipart=0
        npart=0
        return
      endif
      zp=zmax-zp

      fact   = real(ihmax-1) / ( zmax - zmin )
      imi    = max ( 1 , min ( ihmax , nint ( 1. + zp*fact ) ) )

      call ptsort (ihmax,nspec)
      call pt_fld (nspec, npart)
      
      do iang=1, mth
        ipart(:,iang)=imo(1+(iang-1)*mk:iang*mk)
      enddo
      
      end subroutine partition

      subroutine ptsort(iihmax,nnspec)
!
      implicit none
      integer                 :: i, in, iv, iihmax, nnspec
      integer                 :: numv(iihmax), iaddr(iihmax),iorder(nnspec)
! -------------------------------------------------------------------- /
! 1.  occurences per height
!
      numv   = 0
      do i=1, nspec
        numv(imi(i)) = numv(imi(i)) + 1
        end do
!
! -------------------------------------------------------------------- /
! 2.  starting address per height
!
      iaddr(1) = 1
      do i=1, iihmax-1
        iaddr(i+1) = iaddr(i) + numv(i)
      end do
!
! -------------------------------------------------------------------- /
! 3.  order points
!
      do i=1, nnspec
        iv        = imi(i)
        in        = iaddr(iv)
        iorder(i) = in
        iaddr(iv) = in + 1
        end do
!
! -------------------------------------------------------------------- /
! 4.  sort points
!
      do i=1, nnspec
        ind(iorder(i)) = i
        end do
!
      return
!/
!/ end of ptsort ----------------------------------------------------- /
!/
      end subroutine ptsort


!/ ------------------------------------------------------------------- /
      subroutine ptnghb

      implicit none

      integer                 :: n, j, i, k

! -------------------------------------------------------------------- /
! 2.  build map
!
      neigh  = 0
!
! ... base loop
!
      do n = 1, nspec
!
        j      = (n-1) / mk + 1
        i      = n - (j-1) * mk
        k      = 0
!
! ... point at the left(1)
!
        if ( i .ne. 1 ) then
            k           = k + 1
            neigh(k, n) = n - 1
          end if
!
! ... point at the right (2)
!
        if ( i .ne. mk ) then 
            k           = k + 1
            neigh(k, n) = n + 1
          end if
!
! ... point at the bottom(3)
!
        if ( j .ne. 1 ) then
            k           = k + 1
            neigh(k, n) = n - mk
          end if
!
! ... add point at bottom_wrap to top
!
        if ( j .eq. 1 ) then
            k          = k + 1
            neigh(k,n) = nspec - (mk-i)
          end if
!
! ... point at the top(4)
!
        if ( j .ne. mth ) then
            k           = k + 1
            neigh(k, n) = n + mk
          end if
!
! ... add point to top_wrap to bottom
!
         if ( j .eq. mth ) then
             k          = k + 1
             neigh(k,n) = n - (mth-1) * mk
            end if
!
! ... point at the bottom, left(5)
!
        if ( (i.ne.1) .and. (j.ne.1) ) then
            k           = k + 1
            neigh(k, n) = n - mk - 1
          end if
!
! ... point at the bottom, left with wrap.
!
         if ( (i.ne.1) .and. (j.eq.1) ) then
             k          = k + 1
             neigh(k,n) = n - 1 + mk * (mth-1)
           end if
!
! ... point at the bottom, right(6)
!
        if ( (i.ne.mk) .and. (j.ne.1) ) then
            k           = k + 1
            neigh(k, n) = n - mk + 1
          end if
!
! ... point at the bottom, right with wrap
!
        if ( (i.ne.mk) .and. (j.eq.1) ) then
            k           = k + 1
            neigh(k,n) = n + 1 + mk * (mth - 1)
          end  if
!
! ... point at the top, left(7)
!
        if ( (i.ne.1) .and. (j.ne.mth) ) then
            k           = k + 1
            neigh(k, n) = n + mk - 1
          end if
!
! ... point at the top, left with wrap
!
         if ( (i.ne.1) .and. (j.eq.mth) ) then
             k           = k + 1
             neigh(k,n) = n - 1 - (mk) * (mth-1)
           end if
!
! ... point at the top, right(8)
!
        if ( (i.ne.mk) .and. (j.ne.mth) ) then
            k           = k + 1
            neigh(k, n) = n + mk + 1
          end if
!
! ... point at top, right with wrap
!
!
        if ( (i.ne.mk) .and. (j.eq.mth) ) then
            k           = k + 1
            neigh(k,n) = n + 1 - (mk) * (mth-1)
          end if
!
        neigh(9,n) = k
!
        end do
!
      return
!/
!/ end of ptnghb ----------------------------------------------------- /
!/
      end subroutine ptnghb
!/ ------------------------------------------------------------------- /
      subroutine pt_fld (nnspec,npart)
!
      implicit none
      integer, intent(in)     :: nnspec
      integer, intent(out)    :: npart
!/
!/ ------------------------------------------------------------------- /
!/ local parameters
!/
      integer                 :: mask, init, iwshed, imd(nnspec),      &
                                 ic_label, ifict_pixel, m, ih, msave, &
                                 ip, i, ipp, ic_dist, iempty, ippp,   &
                                 jl, jn, ipt, j
      integer                 :: iq(nspec), iq_start, iq_end
      real                    :: zpmax, ep1, diff
! -------------------------------------------------------------------- /
! 0.  initializations
!
      mask        = -2
      init        = -1
      iwshed      =  0
      imo         = init
      ic_label    =  0
      imd         =  0
      ifict_pixel = -100
!
      iq_start    =  1
      iq_end      =  1
!
      zpmax       = maxval ( zp )
!
! -------------------------------------------------------------------- /
! 1.  loop over levels
!
      m      =  1
!
      do ih=1, ihmax
        msave  = m
!
! 1.a pixels at level ih
!
        do
          ip     = ind(m)
          if ( imi(ip) .ne. ih ) exit
!
!     flag the point, if it stays flagge, it is a separate minimum.
!
          imo(ip) = mask
!
!     consider neighbors. if there is neighbor, set distance and add
!     to queue.
!
          do i=1, neigh(9,ip)
            ipp    = neigh(i,ip)
            if ( (imo(ipp).gt.0) .or. (imo(ipp).eq.iwshed) ) then
                imd(ip) = 1
                call fifo_add (ip)
                exit
              end if
            end do
!
          if ( m+1 .gt. nspec ) then
              exit
            else
              m = m + 1
            end if
!
          end do
!
! 1.b process the queue
!
        ic_dist = 1
        call fifo_add (ifict_pixel)
!
        do
          call fifo_first (ip)
!
!     check for end of processing
!
          if ( ip .eq. ifict_pixel ) then
              call fifo_empty (iempty)
              if ( iempty .eq. 1 ) then
                  exit
                else
                  call fifo_add (ifict_pixel)
                  ic_dist = ic_dist + 1
                  call fifo_first (ip)
                end if
            end if
!
!     process queue
!
          do i=1, neigh(9,ip)
            ipp = neigh(i,ip)
!
!     check for labeled watersheds or basins
!
            if ( (imd(ipp).lt.ic_dist) .and. ( (imo(ipp).gt.0) .or.  &
                 (imo(ipp).eq.iwshed))) then
!
                if ( imo(ipp) .gt. 0 ) then
!
                    if ((imo(ip) .eq. mask) .or. (imo(ip) .eq. &
                        iwshed)) then
                        imo(ip) = imo(ipp)
                      else if (imo(ip) .ne. imo(ipp)) then
                        imo(ip) = iwshed
                      end if
!
                  else if (imo(ip) .eq. mask) then
!
                    imo(ip) = iwshed
!
                  end if
!
              else if ( (imo(ipp).eq.mask) .and. (imd(ipp).eq.0) ) then
!
                 imd(ipp) = ic_dist + 1
                 call fifo_add (ipp)
!
              end if
!
            end do
!
          end do
!
! 1.c check for mask values in imo to identify new basins
!
        m = msave
!
        do
          ip     = ind(m)
          if ( imi(ip) .ne. ih ) exit
          imd(ip) = 0
!
          if (imo(ip) .eq. mask) then
!
! ... new label for pixel
!
              ic_label = ic_label + 1
              call fifo_add (ip)
              imo(ip) = ic_label
!
! ... and all connected to it ...
!
              do
                call fifo_empty (iempty)
                if ( iempty .eq. 1 ) exit
                call fifo_first (ipp)
!
                do i=1, neigh(9,ipp)
                  ippp   = neigh(i,ipp)
                  if ( imo(ippp) .eq. mask ) then
                      call fifo_add (ippp)
                      imo(ippp) = ic_label
                    end if
                  end do
!
                end do
!
            end if
!
          if ( m + 1 .gt. nspec ) then
              exit
            else
              m = m + 1
            end if
!
          end do
!
        end do
!
! -------------------------------------------------------------------- /
! 2.  find nearest neighbor of 0 watershed points and replace
!     use original input to check which group to affiliate with 0
!     soring changes first in imd to assure symetry in adjustment.
!
      do j=1, 5
        imd    = imo
        do jl=1 , nspec
          ipt    = -1
          if ( imo(jl) .eq. 0 ) then
              ep1    = zpmax
              do jn=1, neigh (9,jl)
                diff   = abs ( zp(jl) - zp(neigh(jn,jl)))
                if ( (diff.le.ep1) .and. (imo(neigh(jn,jl)).ne.0) ) then
                    ep1    = diff
                    ipt    = jn
                  end if
                end do
              if ( ipt .gt. 0 ) imd(jl) = imo(neigh(ipt,jl))
            end if
          end do
        imo    = imd
        if ( minval(imo) .gt. 0 ) exit
        end do
!
      npart = ic_label
!
      return
!
      contains
!/ ------------------------------------------------------------------- /
      subroutine fifo_add ( iv )
!
!     add point to fifo queue.
!
      integer, intent(in)      :: iv
!
      iq(iq_end) = iv
!
      iq_end = iq_end + 1
      if ( iq_end .gt. nspec ) iq_end = 1
!
      return
      end subroutine
!/ ------------------------------------------------------------------- /
      subroutine fifo_empty ( iempty )
!
!     check if queue is empty.
!
      integer, intent(out)     :: iempty
!
      if ( iq_start .ne. iq_end ) then
        iempty = 0
      else
        iempty = 1
      end if
!
      return
      end subroutine
!/ ------------------------------------------------------------------- /
      subroutine fifo_first ( iv )
!
!     get point out of queue.
!
      integer, intent(out)     :: iv
!
      iv = iq(iq_start)
!
      iq_start = iq_start + 1
      if ( iq_start .gt. nspec ) iq_start = 1
!
      return
      end subroutine
!/
!/ end of pt_fld ----------------------------------------------------- /
!/
      end subroutine pt_fld

      end module specpart
