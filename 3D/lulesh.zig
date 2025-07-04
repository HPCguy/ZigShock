//
//                  Copyright (c) 2010.
//       Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
//                   LLNL-CODE-461231
//                 All rights reserved.
// 
// This file is part of LULESH, Version 1.0.
// Please also read this link -- http://www.opensource.org/licenses/index.php
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the disclaimer below.
// 
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the disclaimer (as noted below)
//      in the documentation and/or other materials provided with the
//      distribution.
// 
//    * Neither the name of the LLNS/LLNL nor the names of its contributors
//      may be used to endorse or promote products derived from this software
//      without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
// THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// 
// Additional BSD Notice
// 
// 1. This notice is required to be provided under our contract with the U.S.
//    Department of Energy (DOE). This work was produced at Lawrence Livermore
//    National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.
// 
// 2. Neither the United States Government nor Lawrence Livermore National
//    Security, LLC nor any of their employees, makes any warranty, express
//    or implied, or assumes any liability or responsibility for the accuracy,
//    completeness, or usefulness of any information, apparatus, product, or
//    process disclosed, or represents that its use would not infringe
//    privately-owned rights.
// 
// 3. Also, reference herein to any specific commercial products, process, or
//    services by trade name, trademark, manufacturer or otherwise does not
//    necessarily itute or imply its endorsement, recommendation, or
//    favoring by the United States Government or Lawrence Livermore National
//    Security, LLC. The views and opinions of authors expressed herein do not
//    necessarily state or reflect those of the United States Government or
//    Lawrence Livermore National Security, LLC, and shall not be used for
//    advertising or product endorsement purposes.
// 

// This application was specifically designed to help in Supercomputer
// procurement decisions.  The goal is to test the vendors compilers
// ability to optimize constructs often found in science applications.
// Examples include inlining of small functions, fusing small loops that
// iterate over the same index space, etc.  Some constructs in the
// source code below are purposefully "dumb" to test optimization capability.
// By carefully analyzing what *does* get optimized, it helps determine the
// quality of the compiler as it relates to scientific software optimization.

const std = @import("std");
const stdout = std.io.getStdOut().writer();
const math = std.math;

// Display timestep information throughout calculation
const LULESH_SHOW_PROGRESS : bool = true;

// Real_t and Index_t are the element types used within arrays.
// The smaller the datatype, the more *potential* there is for
// the compiler to better utilize caches, memory movement,
// and/or vectorization.

const Real_t = f32;
const Index_t = usize;

fn IndexToReal(idx: Index_t) Real_t
{
   return @floatFromInt(idx);
}

// Some compilers need a suffix to differentiate FP data size.
// The following constants help centralize that change.
// A block of scalar initializations are also found in main(),
// where suffix changes could also need to be made.
const ZERO      : Real_t = 0.0;
const SIXTEENTH : Real_t = 0.0625;
const EIGHTH    : Real_t = 0.125;
const QUARTER   : Real_t = 0.25;
const HALF      : Real_t = 0.5;
const ONE       : Real_t = 1.0;
const TWO       : Real_t = 2.0;
const THREE     : Real_t = 3.0;
const FOUR      : Real_t = 4.0;
const SIX       : Real_t = 6.0;
const SEVEN     : Real_t = 7.0;
const EIGHT     : Real_t = 8.0;
const TWELVE    : Real_t = 8.0;

const SIXTYFOUR : Real_t = 64.0;

const Err = enum(u8) { VolumeError = 1, QStopError = 2 } ;

// 2 Boundary Conditions on each of 6 hexahedral faces (12 bits)
const BC = enum(u32) {
     XI_M_SYMM = 0x001,   XI_M_FREE = 0x002 ,  XI_M = 0x003,
     XI_P_SYMM = 0x004,   XI_P_FREE = 0x008 ,  XI_P = 0x00c,
    ETA_M_SYMM = 0x010,  ETA_M_FREE = 0x020,  ETA_M = 0x030,
    ETA_P_SYMM = 0x040,  ETA_P_FREE = 0x080,  ETA_P = 0x0c0,
   ZETA_M_SYMM = 0x100, ZETA_M_FREE = 0x200, ZETA_M = 0x300,
   ZETA_P_SYMM = 0x400, ZETA_P_FREE = 0x800, ZETA_P = 0xc00 } ;

fn BCbits( bits: BC) u32
{
   return @intFromEnum(bits);
}


// *********************************
// * Data structure implementation *
// *********************************

// might want to add access methods so that memory can be
// better managed, as in luleshFT

const Domain = struct {
   // Elem-centered

   nodelist  : [*][8]Index_t,  // elemToNode connectivity

   lxim      : [*]Index_t,     // elem connectivity through face
   lxip      : [*]Index_t,
   letam     : [*]Index_t,
   letap     : [*]Index_t,
   lzetam    : [*]Index_t,
   lzetap    : [*]Index_t,

   elemBC    : [*]u32,         // elem face symm/free-surface flag

   e         : [*]Real_t,      // energy

   p         : [*]Real_t,      // pressure
   q         : [*]Real_t,      // artificial viscosity
   ql        : [*]Real_t,      //   linear    term
   qq        : [*]Real_t,      //   quadratic term

   v         : [*]Real_t,      // relative  volume
   volo      : [*]Real_t,      // reference volume
   delv      : [*]Real_t,      // vnew - v
   vdov      : [*]Real_t,      // volume derivative over volume

   arealg    : [*]Real_t,      // finite elem characteristic length

   ss        : [*]Real_t,      // "sound speed"

   elemMass  : [*]Real_t,      // mass

   // Finite Element temporaries

   dxx       : [*]Real_t,      // principal strains -- temporary
   dyy       : [*]Real_t,
   dzz       : [*]Real_t,

   delv_xi   : [*]Real_t,      // velocity gradient -- temporary
   delv_eta  : [*]Real_t,
   delv_zeta : [*]Real_t,

   delx_xi   : [*]Real_t,      // position gradient -- temporary
   delx_eta  : [*]Real_t,
   delx_zeta : [*]Real_t,

   sigxx     : [*]Real_t,
   sigyy     : [*]Real_t,
   sigzz     : [*]Real_t,

   dvdx      : [*][8]Real_t,
   dvdy      : [*][8]Real_t,
   dvdz      : [*][8]Real_t,

   x8n      : [*][8]Real_t,
   y8n      : [*][8]Real_t,
   z8n      : [*][8]Real_t,

   determ    : [*]Real_t,
   vnew      : [*]Real_t,
   vnewc     : [*]Real_t,

   p_old     : [*]Real_t,
   p_new     : [*]Real_t,
   p_HalfStep: [*]Real_t,
   q_new     : [*]Real_t,
   e_new     : [*]Real_t,
   bvc       : [*]Real_t,
   pbvc      : [*]Real_t,
   work      : [*]Real_t,

   compression : [*]Real_t,
   compressionHalfStep : [*]Real_t,

   // Node-centered

   x         : [*]Real_t,      // coordinates
   y         : [*]Real_t,
   z         : [*]Real_t,

   xd        : [*]Real_t,      // velocities
   yd        : [*]Real_t,
   zd        : [*]Real_t,

   xdd       : [*]Real_t,      // accelerations
   ydd       : [*]Real_t,
   zdd       : [*]Real_t,

   fx        : [*]Real_t,      // forces
   fy        : [*]Real_t,
   fz        : [*]Real_t,

   nodalMass : [*]Real_t,      // mass

   // Boundary nodesets

   symmX     : [*]Index_t,     // Nodes on X symmetry plane
   symmY     : [*]Index_t,     // Nodes on Y symmetry plane
   symmZ     : [*]Index_t,     // Nodes on Z symmetry plane

   // Parameters

   dtfixed   : Real_t,         // fixed time increment
   time      : Real_t,         // current time
   deltatime : Real_t,         // variable time increment
   deltatimemultlb : Real_t,
   deltatimemultub : Real_t,
   stoptime  : Real_t,         // end time for simulation

   u_cut     : Real_t,         // velocity tolerance
   hgcoef    : Real_t,         // hourglass control
   qstop     : Real_t,         // excessive q indicator
   monoq_max_slope : Real_t,
   monoq_limiter_mult : Real_t,
   e_cut     : Real_t,         // energy tolerance
   p_cut     : Real_t,         // pressure tolerance
   q_cut     : Real_t,         // q tolerance
   v_cut     : Real_t,         // relative volume tolerance
   qlc_monoq : Real_t,         // linear term coef for q
   qqc_monoq : Real_t,         // quadratic term coef for q
   qqc       : Real_t,
   eosvmax   : Real_t,
   eosvmin   : Real_t,
   pmin      : Real_t,         // pressure floor
   emin      : Real_t,         // energy floor
   dvovmax   : Real_t,         // maximum allowable volume change
   refdens   : Real_t,         // reference density

   dtcourant : Real_t,         // courant constraint
   dthydro   : Real_t,         // volume change constraint
   dtmax     : Real_t,         // maximum allowable time increment

   cycle     : i32,            // iteration count for simulation

   sizeX     : Index_t,
   sizeY     : Index_t,
   sizeZ     : Index_t,
   numElem   : Index_t,

   numNode   : Index_t
} ;

fn TimeIncrement(domain: *Domain) void
{
   var targetdt: Real_t = domain.stoptime - domain.time;
   var newdt: Real_t = domain.deltatime;

   if ((domain.dtfixed <= ZERO) and (domain.cycle != 0)) {
      const olddt: Real_t = domain.deltatime;

      // This will require a reduction in parallel
      newdt = 1.0e+20;
      if (domain.dtcourant < newdt) {
         newdt = domain.dtcourant / TWO;
      }
      if (domain.dthydro < newdt) {
         newdt = domain.dthydro * TWO / THREE;
      }

      const ratio: Real_t = newdt / olddt;
      if (ratio >= ONE) {
         if (ratio < domain.deltatimemultlb) {
            newdt = olddt;
         }
         else if (ratio > domain.deltatimemultub) {
            newdt = olddt*domain.deltatimemultub;
         }
      }

      if (newdt > domain.dtmax) {
         newdt = domain.dtmax;
      }
   }

   // TRY TO PREVENT VERY SMALL SCALING ON THE NEXT CYCLE
   if ((targetdt > newdt) and
       (targetdt < (FOUR * newdt / THREE)) ) {
      targetdt = TWO * newdt / THREE;
   }

   if (targetdt < newdt) {
      newdt = targetdt;
   }

   domain.time += newdt;

   domain.deltatime = newdt;

   domain.cycle += 1;
}

fn InitStressTermsForElems(p: [*]Real_t, q: [*]Real_t,
                           sigxx: [*]Real_t, sigyy: [*]Real_t,
                           sigzz: [*]Real_t, numElem: Index_t) void
{
   var idx: Index_t = 0;
   while ( idx < numElem) : ( idx += 1 ) {
      const stress = - p[idx] - q[idx];
      sigxx[idx] = stress;
      sigyy[idx] = stress;
      sigzz[idx] = stress;
   }
}

fn CalcElemShapeFunctionDerivatives(x: [8]Real_t, y: [8]Real_t, z: [8]Real_t,
                                    B: [][8]Real_t) Real_t
{
   const x0 = x[0];    const x1 = x[1];
   const x2 = x[2];    const x3 = x[3];
   const x4 = x[4];    const x5 = x[5];
   const x6 = x[6];    const x7 = x[7];

   const t1 = (x6-x0); const t2 = (x5-x3);
   const t3 = (x7-x1); const t4 = (x4-x2);

   const fjxxi = EIGHTH * ( t1 + t2 - t3 - t4 );
   const fjxet = EIGHTH * ( t1 - t2 + t3 - t4 );
   const fjxze = EIGHTH * ( t1 + t2 + t3 + t4 );

   const y0 = y[0];    const y1 = y[1];
   const y2 = y[2];    const y3 = y[3];
   const y4 = y[4];    const y5 = y[5];
   const y6 = y[6];    const y7 = y[7];

   const t1a = (y6-y0); const t2a = (y5-y3);
   const t3a = (y7-y1); const t4a = (y4-y2);

   const fjyxi = EIGHTH * ( t1a + t2a - t3a - t4a );
   const fjyet = EIGHTH * ( t1a - t2a + t3a - t4a );
   const fjyze = EIGHTH * ( t1a + t2a + t3a + t4a );

   const z0 = z[0];    const z1 = z[1];
   const z2 = z[2];    const z3 = z[3];
   const z4 = z[4];    const z5 = z[5];
   const z6 = z[6];    const z7 = z[7];

   const t1b = (z6-z0); const t2b = (z5-z3);
   const t3b = (z7-z1); const t4b = (z4-z2);

   const fjzxi = EIGHTH * ( t1b + t2b - t3b - t4b );
   const fjzet = EIGHTH * ( t1b - t2b + t3b - t4b );
   const fjzze = EIGHTH * ( t1b + t2b + t3b + t4b );

   // compute cofactors
   const cjxxi =    (fjyet * fjzze) - (fjzet * fjyze);
   const cjxet =  - (fjyxi * fjzze) + (fjzxi * fjyze);
   const cjxze =    (fjyxi * fjzet) - (fjzxi * fjyet);

   const cjyxi =  - (fjxet * fjzze) + (fjzet * fjxze);
   const cjyet =    (fjxxi * fjzze) - (fjzxi * fjxze);
   const cjyze =  - (fjxxi * fjzet) + (fjzxi * fjxet);

   const cjzxi =    (fjxet * fjyze) - (fjyet * fjxze);
   const cjzet =  - (fjxxi * fjyze) + (fjyxi * fjxze);
   const cjzze =    (fjxxi * fjyet) - (fjyxi * fjxet);

   // calculate partials :
   // this need only be done for l = 0,1,2,3   since , by symmetry ,
   // (6,7,4,5) = - (0,1,2,3) .
   //
   B[0][0] =   -  cjxxi  -  cjxet  -  cjxze;  B[0][6] = -B[0][0];
   B[0][1] =      cjxxi  -  cjxet  -  cjxze;  B[0][7] = -B[0][1];
   B[0][2] =      cjxxi  +  cjxet  -  cjxze;  B[0][4] = -B[0][2];
   B[0][3] =   -  cjxxi  +  cjxet  -  cjxze;  B[0][5] = -B[0][3];

   B[1][0] =   -  cjyxi  -  cjyet  -  cjyze;  B[1][6] = -B[1][0];
   B[1][1] =      cjyxi  -  cjyet  -  cjyze;  B[1][7] = -B[1][1];
   B[1][2] =      cjyxi  +  cjyet  -  cjyze;  B[1][4] = -B[1][2];
   B[1][3] =   -  cjyxi  +  cjyet  -  cjyze;  B[1][5] = -B[1][3];

   B[2][0] =   -  cjzxi  -  cjzet  -  cjzze;  B[2][6] = -B[2][0];
   B[2][1] =      cjzxi  -  cjzet  -  cjzze;  B[2][7] = -B[2][1];
   B[2][2] =      cjzxi  +  cjzet  -  cjzze;  B[2][4] = -B[2][2];
   B[2][3] =   -  cjzxi  +  cjzet  -  cjzze;  B[2][5] = -B[2][3];

   // calculate jacobian determinant (volume)
   return 8.0 * ( fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);
}

fn SumElemFaceNormal(normalX0: *Real_t, normalY0: *Real_t, normalZ0: *Real_t,
                     normalX1: *Real_t, normalY1: *Real_t, normalZ1: *Real_t,
                     normalX2: *Real_t, normalY2: *Real_t, normalZ2: *Real_t,
                     normalX3: *Real_t, normalY3: *Real_t, normalZ3: *Real_t,
                     x0: Real_t, y0: Real_t, z0: Real_t,
                     x1: Real_t, y1: Real_t, z1: Real_t,
                     x2: Real_t, y2: Real_t, z2: Real_t,
                     x3: Real_t, y3: Real_t, z3: Real_t) void
{
   const bisectX0 = x3 + x2 - x1 - x0;
   const bisectY0 = y3 + y2 - y1 - y0;
   const bisectZ0 = z3 + z2 - z1 - z0;
   const bisectX1 = x2 + x1 - x3 - x0;
   const bisectY1 = y2 + y1 - y3 - y0;
   const bisectZ1 = z2 + z1 - z3 - z0;
   const areaX = SIXTEENTH * (bisectY0 * bisectZ1 - bisectZ0 * bisectY1);
   const areaY = SIXTEENTH * (bisectZ0 * bisectX1 - bisectX0 * bisectZ1);
   const areaZ = SIXTEENTH * (bisectX0 * bisectY1 - bisectY0 * bisectX1);

   normalX0.* += areaX;
   normalX1.* += areaX;
   normalX2.* += areaX;
   normalX3.* += areaX;

   normalY0.* += areaY;
   normalY1.* += areaY;
   normalY2.* += areaY;
   normalY3.* += areaY;

   normalZ0.* += areaZ;
   normalZ1.* += areaZ;
   normalZ2.* += areaZ;
   normalZ3.* += areaZ;
}

fn CalcElemNodeNormals(pfx: []Real_t, pfy: []Real_t, pfz: []Real_t,
                       x: [8]Real_t, y: [8]Real_t, z: [8]Real_t) void
{
   // pfx = .{ ZERO } ** 8;
   // pfy = .{ ZERO } ** 8;
   // pfz = .{ ZERO } ** 8;
   var i: Index_t = 0;
   while ( i < 8 ) : ( i += 1 ) {
      pfx[i] = ZERO;
      pfy[i] = ZERO;
      pfz[i] = ZERO;
   }
   // evaluate face one: nodes 0, 1, 2, 3
   SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                  &pfx[1], &pfy[1], &pfz[1],
                  &pfx[2], &pfy[2], &pfz[2],
                  &pfx[3], &pfy[3], &pfz[3],
                  x[0], y[0], z[0], x[1], y[1], z[1],
                  x[2], y[2], z[2], x[3], y[3], z[3]);
   // evaluate face two: nodes 0, 4, 5, 1
   SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                  &pfx[4], &pfy[4], &pfz[4],
                  &pfx[5], &pfy[5], &pfz[5],
                  &pfx[1], &pfy[1], &pfz[1],
                  x[0], y[0], z[0], x[4], y[4], z[4],
                  x[5], y[5], z[5], x[1], y[1], z[1]);
   // evaluate face three: nodes 1, 5, 6, 2
   SumElemFaceNormal(&pfx[1], &pfy[1], &pfz[1],
                  &pfx[5], &pfy[5], &pfz[5],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[2], &pfy[2], &pfz[2],
                  x[1], y[1], z[1], x[5], y[5], z[5],
                  x[6], y[6], z[6], x[2], y[2], z[2]);
   // evaluate face four: nodes 2, 6, 7, 3
   SumElemFaceNormal(&pfx[2], &pfy[2], &pfz[2],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[3], &pfy[3], &pfz[3],
                  x[2], y[2], z[2], x[6], y[6], z[6],
                  x[7], y[7], z[7], x[3], y[3], z[3]);
   // evaluate face five: nodes 3, 7, 4, 0
   SumElemFaceNormal(&pfx[3], &pfy[3], &pfz[3],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[4], &pfy[4], &pfz[4],
                  &pfx[0], &pfy[0], &pfz[0],
                  x[3], y[3], z[3], x[7], y[7], z[7],
                  x[4], y[4], z[4], x[0], y[0], z[0]);
   // evaluate face six: nodes 4, 7, 6, 5
   SumElemFaceNormal(&pfx[4], &pfy[4], &pfz[4],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[5], &pfy[5], &pfz[5],
                  x[4], y[4], z[4], x[7], y[7], z[7],
                  x[6], y[6], z[6], x[5], y[5], z[5]);
}

fn SumElemStressesToNodeForces(B: [3][8]Real_t, stress_xx: Real_t,
                               stress_yy: Real_t, stress_zz: Real_t,
                               fx: []Real_t, fy: []Real_t, fz: []Real_t)
                               void
{
  // fx.data = - ( stress_xx * B[0].data );
  // fy.data = - ( stress_yy * B[1].data );
  // fz.data = - ( stress_zz * B[2].data );
  var i: Index_t = 0;
  while ( i < 8 ) : ( i += 1 ) {
     fx[i] = - ( stress_xx * B[0][i] );
     fy[i] = - ( stress_yy * B[1][i] );
     fz[i] = - ( stress_zz * B[2][i] );
  }
}

fn GatherNodes(elemNodes: [8]Index_t, x: [*]Real_t, y: [*]Real_t, z: [*]Real_t,
               x_local: []Real_t, y_local: []Real_t, z_local: []Real_t) void
{
  var lnode: Index_t = 0;
  while ( lnode < 8 ) : ( lnode += 1 ) {
    const gnode = elemNodes[lnode];
    x_local[lnode] = x[gnode];
    y_local[lnode] = y[gnode];
    z_local[lnode] = z[gnode];
  }
}

fn SumForce(elemNodes: [8]Index_t, fx: [*]Real_t, fy: [*]Real_t, fz: [*]Real_t,
            fx_local: [8]Real_t, fy_local: [8]Real_t, fz_local: [8]Real_t) void
{
  var lnode: Index_t = 0;
  while ( lnode < 8 ) : ( lnode += 1 ) {
    const gnode = elemNodes[lnode];
    fx[gnode] += fx_local[lnode];
    fy[gnode] += fy_local[lnode];
    fz[gnode] += fz_local[lnode];
  }
}

fn IntegrateStressForElems(nodelist: [*][8]Index_t,
                           x: [*]Real_t,  y: [*]Real_t, z: [*]Real_t,
                           fx: [*]Real_t, fy: [*]Real_t, fz: [*]Real_t,
                           sigxx: [*]Real_t, sigyy: [*]Real_t, sigzz: [*]Real_t,
                           determ: [*]Real_t, numElem: Index_t) void
{
  // loop over all elements
  var k: Index_t = 0;
  while ( k < numElem ) : ( k += 1 ) {
    const elemNodes = nodelist[k];

    var B: [3][8]Real_t = undefined; // shape function derivatives

    {
      var x_local: [8]Real_t = undefined;
      var y_local: [8]Real_t = undefined;
      var z_local: [8]Real_t = undefined;

      // get coordinates from global arrays and copy into local arrays.
      GatherNodes(elemNodes, x, y, z, &x_local, &y_local, &z_local);

      // Volume calculation involves extra work for numerical consistency.
      determ[k] =
         CalcElemShapeFunctionDerivatives(x_local, y_local, z_local, &B);

      CalcElemNodeNormals( &B[0] , &B[1], &B[2],
                           x_local, y_local, z_local );
    }

    {
      var fx_local: [8]Real_t = undefined;
      var fy_local: [8]Real_t = undefined;
      var fz_local: [8]Real_t = undefined;

      SumElemStressesToNodeForces( B, sigxx[k], sigyy[k], sigzz[k],
                                   &fx_local, &fy_local, &fz_local );

      // sum nodal force contributions to global force arrray.
      SumForce(elemNodes, fx, fy, fz, fx_local, fy_local, fz_local);
    }
  }
}

fn CollectDomainNodesToElemNodes(x: [*]Real_t, y: [*]Real_t, z: [*]Real_t,
                                   elemToNode: [8]Index_t, elemX: []Real_t,
                                   elemY: []Real_t, elemZ: []Real_t) void
{
   const nd0i = elemToNode[0];
   const nd1i = elemToNode[1];
   const nd2i = elemToNode[2];
   const nd3i = elemToNode[3];
   const nd4i = elemToNode[4];
   const nd5i = elemToNode[5];
   const nd6i = elemToNode[6];
   const nd7i = elemToNode[7];

   elemX[0] = x[nd0i];
   elemX[1] = x[nd1i];
   elemX[2] = x[nd2i];
   elemX[3] = x[nd3i];
   elemX[4] = x[nd4i];
   elemX[5] = x[nd5i];
   elemX[6] = x[nd6i];
   elemX[7] = x[nd7i];

   elemY[0] = y[nd0i];
   elemY[1] = y[nd1i];
   elemY[2] = y[nd2i];
   elemY[3] = y[nd3i];
   elemY[4] = y[nd4i];
   elemY[5] = y[nd5i];
   elemY[6] = y[nd6i];
   elemY[7] = y[nd7i];

   elemZ[0] = z[nd0i];
   elemZ[1] = z[nd1i];
   elemZ[2] = z[nd2i];
   elemZ[3] = z[nd3i];
   elemZ[4] = z[nd4i];
   elemZ[5] = z[nd5i];
   elemZ[6] = z[nd6i];
   elemZ[7] = z[nd7i];

}

fn VoluDer(x0: Real_t, x1: Real_t, x2: Real_t,
             x3: Real_t, x4: Real_t, x5: Real_t,
             y0: Real_t, y1: Real_t, y2: Real_t,
             y3: Real_t, y4: Real_t, y5: Real_t,
             z0: Real_t, z1: Real_t, z2: Real_t,
             z3: Real_t, z4: Real_t, z5: Real_t,
             dvdx: *Real_t, dvdy: *Real_t, dvdz: *Real_t) void
{
   const twelfth = ONE / TWELVE;

   dvdx.* =
     ((y1 + y2) * (z0 + z1) - (y0 + y1) * (z1 + z2) +
      (y0 + y4) * (z3 + z4) - (y3 + y4) * (z0 + z4) -
      (y2 + y5) * (z3 + z5) + (y3 + y5) * (z2 + z5)) * twelfth;
   dvdy.* =
     (- (x1 + x2) * (z0 + z1) + (x0 + x1) * (z1 + z2) -
      (x0 + x4) * (z3 + z4) + (x3 + x4) * (z0 + z4) +
      (x2 + x5) * (z3 + z5) - (x3 + x5) * (z2 + z5)) * twelfth;

   dvdz.* =
     (- (y1 + y2) * (x0 + x1) + (y0 + y1) * (x1 + x2) -
      (y0 + y4) * (x3 + x4) + (y3 + y4) * (x0 + x4) +
      (y2 + y5) * (x3 + x5) - (y3 + y5) * (x2 + x5)) * twelfth;
}

fn CalcElemVolumeDerivative(dvdx: []Real_t, dvdy: []Real_t, dvdz: []Real_t,
                            x: [8]Real_t, y: [8]Real_t, z: [8]Real_t) void
{
   VoluDer(x[1], x[2], x[3], x[4], x[5], x[7],
           y[1], y[2], y[3], y[4], y[5], y[7],
           z[1], z[2], z[3], z[4], z[5], z[7],
           &dvdx[0], &dvdy[0], &dvdz[0]);
   VoluDer(x[0], x[1], x[2], x[7], x[4], x[6],
           y[0], y[1], y[2], y[7], y[4], y[6],
           z[0], z[1], z[2], z[7], z[4], z[6],
           &dvdx[3], &dvdy[3], &dvdz[3]);
   VoluDer(x[3], x[0], x[1], x[6], x[7], x[5],
           y[3], y[0], y[1], y[6], y[7], y[5],
           z[3], z[0], z[1], z[6], z[7], z[5],
           &dvdx[2], &dvdy[2], &dvdz[2]);
   VoluDer(x[2], x[3], x[0], x[5], x[6], x[4],
           y[2], y[3], y[0], y[5], y[6], y[4],
           z[2], z[3], z[0], z[5], z[6], z[4],
           &dvdx[1], &dvdy[1], &dvdz[1]);
   VoluDer(x[7], x[6], x[5], x[0], x[3], x[1],
           y[7], y[6], y[5], y[0], y[3], y[1],
           z[7], z[6], z[5], z[0], z[3], z[1],
           &dvdx[4], &dvdy[4], &dvdz[4]);
   VoluDer(x[4], x[7], x[6], x[1], x[0], x[2],
           y[4], y[7], y[6], y[1], y[0], y[2],
           z[4], z[7], z[6], z[1], z[0], z[2],
           &dvdx[5], &dvdy[5], &dvdz[5]);
   VoluDer(x[5], x[4], x[7], x[2], x[1], x[3],
           y[5], y[4], y[7], y[2], y[1], y[3],
           z[5], z[4], z[7], z[2], z[1], z[3],
           &dvdx[6], &dvdy[6], &dvdz[6]);
   VoluDer(x[6], x[5], x[4], x[3], x[2], x[0],
           y[6], y[5], y[4], y[3], y[2], y[0],
           z[6], z[5], z[4], z[3], z[2], z[0],
           &dvdx[7], &dvdy[7], &dvdz[7]);
}

fn CalcElemFBHourglassForce(xd: [8]Real_t, yd: [8]Real_t, zd: [8]Real_t,
                            hourgam: [][8]Real_t, coefficient: Real_t,
                            hgfx: []Real_t, hgfy: []Real_t, hgfz: []Real_t)
                            void
{
   const k00 = 0;
   const k01 = 1;
   const k02 = 2;
   const k03 = 3;

   const h00 =
      hourgam[k00][0] * xd[0] + hourgam[k00][1] * xd[1] +
      hourgam[k00][2] * xd[2] + hourgam[k00][3] * xd[3] +
      hourgam[k00][4] * xd[4] + hourgam[k00][5] * xd[5] +
      hourgam[k00][6] * xd[6] + hourgam[k00][7] * xd[7];

   const h01 =
      hourgam[k01][0] * xd[0] + hourgam[k01][1] * xd[1] +
      hourgam[k01][2] * xd[2] + hourgam[k01][3] * xd[3] +
      hourgam[k01][4] * xd[4] + hourgam[k01][5] * xd[5] +
      hourgam[k01][6] * xd[6] + hourgam[k01][7] * xd[7];

   const h02 =
      hourgam[k02][0] * xd[0] + hourgam[k02][1] * xd[1] +
      hourgam[k02][2] * xd[2] + hourgam[k02][3] * xd[3] +
      hourgam[k02][4] * xd[4] + hourgam[k02][5] * xd[5] +
      hourgam[k02][6] * xd[6] + hourgam[k02][7] * xd[7];

   const h03 =
      hourgam[k03][0] * xd[0] + hourgam[k03][1] * xd[1] +
      hourgam[k03][2] * xd[2] + hourgam[k03][3] * xd[3] +
      hourgam[k03][4] * xd[4] + hourgam[k03][5] * xd[5] +
      hourgam[k03][6] * xd[6] + hourgam[k03][7] * xd[7];

   hgfx[0] = coefficient *
      (hourgam[k00][0] * h00 + hourgam[k01][0] * h01 +
       hourgam[k02][0] * h02 + hourgam[k03][0] * h03);

   hgfx[1] = coefficient *
      (hourgam[k00][1] * h00 + hourgam[k01][1] * h01 +
       hourgam[k02][1] * h02 + hourgam[k03][1] * h03);

   hgfx[2] = coefficient *
      (hourgam[k00][2] * h00 + hourgam[k01][2] * h01 +
       hourgam[k02][2] * h02 + hourgam[k03][2] * h03);

   hgfx[3] = coefficient *
      (hourgam[k00][3] * h00 + hourgam[k01][3] * h01 +
       hourgam[k02][3] * h02 + hourgam[k03][3] * h03);

   hgfx[4] = coefficient *
      (hourgam[k00][4] * h00 + hourgam[k01][4] * h01 +
       hourgam[k02][4] * h02 + hourgam[k03][4] * h03);

   hgfx[5] = coefficient *
      (hourgam[k00][5] * h00 + hourgam[k01][5] * h01 +
       hourgam[k02][5] * h02 + hourgam[k03][5] * h03);

   hgfx[6] = coefficient *
      (hourgam[k00][6] * h00 + hourgam[k01][6] * h01 +
       hourgam[k02][6] * h02 + hourgam[k03][6] * h03);

   hgfx[7] = coefficient *
      (hourgam[k00][7] * h00 + hourgam[k01][7] * h01 +
       hourgam[k02][7] * h02 + hourgam[k03][7] * h03);

   const h00a =
      hourgam[k00][0] * yd[0] + hourgam[k00][1] * yd[1] +
      hourgam[k00][2] * yd[2] + hourgam[k00][3] * yd[3] +
      hourgam[k00][4] * yd[4] + hourgam[k00][5] * yd[5] +
      hourgam[k00][6] * yd[6] + hourgam[k00][7] * yd[7];

   const h01a =
      hourgam[k01][0] * yd[0] + hourgam[k01][1] * yd[1] +
      hourgam[k01][2] * yd[2] + hourgam[k01][3] * yd[3] +
      hourgam[k01][4] * yd[4] + hourgam[k01][5] * yd[5] +
      hourgam[k01][6] * yd[6] + hourgam[k01][7] * yd[7];

   const h02a =
      hourgam[k02][0] * yd[0] + hourgam[k02][1] * yd[1] +
      hourgam[k02][2] * yd[2] + hourgam[k02][3] * yd[3] +
      hourgam[k02][4] * yd[4] + hourgam[k02][5] * yd[5] +
      hourgam[k02][6] * yd[6] + hourgam[k02][7] * yd[7];

   const h03a =
      hourgam[k03][0] * yd[0] + hourgam[k03][1] * yd[1] +
      hourgam[k03][2] * yd[2] + hourgam[k03][3] * yd[3] +
      hourgam[k03][4] * yd[4] + hourgam[k03][5] * yd[5] +
      hourgam[k03][6] * yd[6] + hourgam[k03][7] * yd[7];

   hgfy[0] = coefficient *
      (hourgam[k00][0] * h00a + hourgam[k01][0] * h01a +
       hourgam[k02][0] * h02a + hourgam[k03][0] * h03a);

   hgfy[1] = coefficient *
      (hourgam[k00][1] * h00a + hourgam[k01][1] * h01a +
       hourgam[k02][1] * h02a + hourgam[k03][1] * h03a);

   hgfy[2] = coefficient *
      (hourgam[k00][2] * h00a + hourgam[k01][2] * h01a +
       hourgam[k02][2] * h02a + hourgam[k03][2] * h03a);

   hgfy[3] = coefficient *
      (hourgam[k00][3] * h00a + hourgam[k01][3] * h01 +
       hourgam[k02][3] * h02a + hourgam[k03][3] * h03);

   hgfy[4] = coefficient *
      (hourgam[k00][4] * h00a + hourgam[k01][4] * h01a +
       hourgam[k02][4] * h02a + hourgam[k03][4] * h03a);

   hgfy[5] = coefficient *
      (hourgam[k00][5] * h00a + hourgam[k01][5] * h01a +
       hourgam[k02][5] * h02a + hourgam[k03][5] * h03a);

   hgfy[6] = coefficient *
      (hourgam[k00][6] * h00a + hourgam[k01][6] * h01a +
       hourgam[k02][6] * h02a + hourgam[k03][6] * h03a);

   hgfy[7] = coefficient *
      (hourgam[k00][7] * h00a + hourgam[k01][7] * h01a +
       hourgam[k02][7] * h02a + hourgam[k03][7] * h03a);

   const h00b =
      hourgam[k00][0] * zd[0] + hourgam[k00][1] * zd[1] +
      hourgam[k00][2] * zd[2] + hourgam[k00][3] * zd[3] +
      hourgam[k00][4] * zd[4] + hourgam[k00][5] * zd[5] +
      hourgam[k00][6] * zd[6] + hourgam[k00][7] * zd[7];

   const h01b =
      hourgam[k01][0] * zd[0] + hourgam[k01][1] * zd[1] +
      hourgam[k01][2] * zd[2] + hourgam[k01][3] * zd[3] +
      hourgam[k01][4] * zd[4] + hourgam[k01][5] * zd[5] +
      hourgam[k01][6] * zd[6] + hourgam[k01][7] * zd[7];

   const h02b =
      hourgam[k02][0] * zd[0] + hourgam[k02][1] * zd[1] +
      hourgam[k02][2] * zd[2] + hourgam[k02][3] * zd[3] +
      hourgam[k02][4] * zd[4] + hourgam[k02][5] * zd[5] +
      hourgam[k02][6] * zd[6] + hourgam[k02][7] * zd[7];

   const h03b =
      hourgam[k03][0] * zd[0] + hourgam[k03][1] * zd[1] +
      hourgam[k03][2] * zd[2] + hourgam[k03][3] * zd[3] +
      hourgam[k03][4] * zd[4] + hourgam[k03][5] * zd[5] +
      hourgam[k03][6] * zd[6] + hourgam[k03][7] * zd[7];

   hgfz[0] = coefficient *
      (hourgam[k00][0] * h00b + hourgam[k01][0] * h01b +
       hourgam[k02][0] * h02b + hourgam[k03][0] * h03b);

   hgfz[1] = coefficient *
      (hourgam[k00][1] * h00b + hourgam[k01][1] * h01b +
       hourgam[k02][1] * h02b + hourgam[k03][1] * h03b);

   hgfz[2] = coefficient *
      (hourgam[k00][2] * h00b + hourgam[k01][2] * h01b +
       hourgam[k02][2] * h02b + hourgam[k03][2] * h03b);

   hgfz[3] = coefficient *
      (hourgam[k00][3] * h00b + hourgam[k01][3] * h01b +
       hourgam[k02][3] * h02b + hourgam[k03][3] * h03b);

   hgfz[4] = coefficient *
      (hourgam[k00][4] * h00b + hourgam[k01][4] * h01b +
       hourgam[k02][4] * h02b + hourgam[k03][4] * h03b);

   hgfz[5] = coefficient *
      (hourgam[k00][5] * h00b + hourgam[k01][5] * h01b +
       hourgam[k02][5] * h02b + hourgam[k03][5] * h03b);

   hgfz[6] = coefficient *
      (hourgam[k00][6] * h00b + hourgam[k01][6] * h01b +
       hourgam[k02][6] * h02b + hourgam[k03][6] * h03b);

   hgfz[7] = coefficient *
      (hourgam[k00][7] * h00b + hourgam[k01][7] * h01b +
       hourgam[k02][7] * h02b + hourgam[k03][7] * h03b);
}

const gammaa: [4][8]Real_t =
.{
   .{  ONE,  ONE, -ONE, -ONE, -ONE, -ONE, ONE,  ONE },
   .{  ONE, -ONE, -ONE,  ONE, -ONE,  ONE, ONE, -ONE },
   .{  ONE, -ONE,  ONE, -ONE,  ONE, -ONE, ONE, -ONE },
   .{ -ONE,  ONE, -ONE,  ONE,  ONE, -ONE, ONE, -ONE }
} ;

fn FBKernel(x8ni: [8]Real_t, y8ni: [8]Real_t, z8ni: [8]Real_t,
              dvdxi: [8]Real_t, dvdyi: [8]Real_t, dvdzi: [8]Real_t,
              hourgam: [][8]Real_t, volinv: Real_t) void
{
   var k1: Index_t = 0;
   while( k1 < 4 ) : ( k1 += 1) {
      const gami = gammaa[k1];
      var hg = hourgam[k1];

      const hourmodx =
         x8ni[0] * gami[0] + x8ni[1] * gami[1] +
         x8ni[2] * gami[2] + x8ni[3] * gami[3] +
         x8ni[4] * gami[4] + x8ni[5] * gami[5] +
         x8ni[6] * gami[6] + x8ni[7] * gami[7];

      const hourmody =
         y8ni[0] * gami[0] + y8ni[1] * gami[1] +
         y8ni[2] * gami[2] + y8ni[3] * gami[3] +
         y8ni[4] * gami[4] + y8ni[5] * gami[5] +
         y8ni[6] * gami[6] + y8ni[7] * gami[7];

      const hourmodz =
         z8ni[0] * gami[0] + z8ni[1] * gami[1] +
         z8ni[2] * gami[2] + z8ni[3] * gami[3] +
         z8ni[4] * gami[4] + z8ni[5] * gami[5] +
         z8ni[6] * gami[6] + z8ni[7] * gami[7];

      hg[0] = gami[0] -  volinv*(dvdxi[0] * hourmodx +
                                 dvdyi[0] * hourmody +
                                 dvdzi[0] * hourmodz );

      hg[1] = gami[1] -  volinv*(dvdxi[1] * hourmodx +
                                 dvdyi[1] * hourmody +
                                 dvdzi[1] * hourmodz );

      hg[2] = gami[2] -  volinv*(dvdxi[2] * hourmodx +
                                 dvdyi[2] * hourmody +
                                 dvdzi[2] * hourmodz );

      hg[3] = gami[3] -  volinv*(dvdxi[3] * hourmodx +
                                 dvdyi[3] * hourmody +
                                 dvdzi[3] * hourmodz );

      hg[4] = gami[4] -  volinv*(dvdxi[4] * hourmodx +
                                 dvdyi[4] * hourmody +
                                 dvdzi[4] * hourmodz );

      hg[5] = gami[5] -  volinv*(dvdxi[5] * hourmodx +
                                 dvdyi[5] * hourmody +
                                 dvdzi[5] * hourmodz );

      hg[6] = gami[6] -  volinv*(dvdxi[6] * hourmodx +
                                 dvdyi[6] * hourmody +
                                 dvdzi[6] * hourmodz );

      hg[7] = gami[7] -  volinv*(dvdxi[7] * hourmodx +
                                 dvdyi[7] * hourmody +
                                 dvdzi[7] * hourmodz );
   }
}

fn CalcFBHourglassForceForElems(nodelist: [*][8]Index_t,
                                  ss: [*]Real_t, elemMass: [*]Real_t,
                                  xd: [*]Real_t, yd: [*]Real_t, zd: [*]Real_t,
                                  fx: [*]Real_t, fy: [*]Real_t, fz: [*]Real_t,
                                  determ: [*]Real_t,
                                  x8n: [*][8]Real_t, y8n: [*][8]Real_t,
                                  z8n: [*][8]Real_t, dvdx: [*][8]Real_t,
                                  dvdy: [*][8]Real_t, dvdz: [*][8]Real_t,
                                  hourg: Real_t, numElem: Index_t) void
{
   // Calculates the Flanagan-Belytschko anti-hourglass force.

   var k2: Index_t = 0;
   while ( k2 < numElem ) : ( k2 += 1 ) {
      const elemToNode = nodelist[k2];

      // compute the hourglass modes */

      var hourgam: [4][8]Real_t = undefined;

      FBKernel( x8n[k2],  y8n[k2],  z8n[k2],
               dvdx[k2], dvdy[k2], dvdz[k2],
               &hourgam, ONE/determ[k2]);

      // compute forces

      const   ss1 = ss[k2];
      const mass1 = elemMass[k2];
      const volume13 = math.cbrt(determ[k2]);

      var  xd1: [8]Real_t = undefined;
      var  yd1: [8]Real_t = undefined;
      var  zd1: [8]Real_t = undefined;
      var hgfx: [8]Real_t = undefined;
      var hgfy: [8]Real_t = undefined;
      var hgfz: [8]Real_t = undefined;

      const coefficient = - hourg * 0.01 * ss1 * mass1 / volume13;

      GatherNodes(elemToNode, xd, yd, zd, &xd1, &yd1, &zd1);

      CalcElemFBHourglassForce(xd1, yd1, zd1, &hourgam,
                               coefficient, &hgfx, &hgfy, &hgfz);

      SumForce(elemToNode, fx, fy, fz, hgfx, hgfy, hgfz);
   }
}

fn CopyBlock(dst1: []Real_t, dst2: []Real_t, dst3: []Real_t,
             src1: [8]Real_t, src2: [8]Real_t, src3: [8]Real_t) void
{
  var i: Index_t = 0;
  while ( i < 8 ) : ( i += 1 ) {
    dst1[i] = src1[i];
    dst2[i] = src2[i];
    dst3[i] = src3[i];
  }
}

fn CalcHourglassControlForElems(domain: *Domain,
                                  determ: [*]Real_t, hgcoef: Real_t) void
{
   const numElem = domain.numElem;

   const dvdx = domain.dvdx;
   const dvdy = domain.dvdy;
   const dvdz = domain.dvdz;
   const x8n  = domain.x8n;
   const y8n  = domain.y8n;
   const z8n  = domain.z8n;

   var idx: Index_t = 0;
   while ( idx < numElem ) : ( idx += 1 ) {
      var  x1: [8]Real_t = undefined;
      var  y1: [8]Real_t = undefined;
      var  z1: [8]Real_t = undefined;
      var pfx: [8]Real_t = undefined;
      var pfy: [8]Real_t = undefined;
      var pfz: [8]Real_t = undefined;

      const elemToNode = domain.nodelist[idx];
      CollectDomainNodesToElemNodes(domain.x, domain.y, domain.z,
                                    elemToNode, &x1, &y1, &z1);

      CalcElemVolumeDerivative(&pfx, &pfy, &pfz, x1, y1, z1);

      // load into temporary storage for FB Hour Glass control
      CopyBlock(&dvdx[idx], &dvdy[idx], &dvdz[idx], pfx, pfy, pfz);
      CopyBlock(&x8n[idx], &y8n[idx], &z8n[idx], x1, y1, z1);

      // calculate absolute volume
      determ[idx] = domain.volo[idx] * domain.v[idx];

      //  check for negative volumes
      if ( domain.v[idx] <= ZERO )
         std.process.exit(@intFromEnum(Err.VolumeError));
   }

   if ( hgcoef > ZERO ) {
      CalcFBHourglassForceForElems( domain.nodelist,
                                    domain.ss, domain.elemMass,
                                    domain.xd, domain.yd, domain.zd,
                                    domain.fx, domain.fy, domain.fz,
                                    determ, x8n, y8n, z8n, dvdx, dvdy, dvdz,
                                    hgcoef, numElem);
   }

   return ;
}

fn VolErr1(determ: [*]Real_t, numElem: Index_t) bool
{
  var k: Index_t = 0;
  while (k < numElem) : ( k += 1 ) {
    if ( determ[k] <= ZERO ) return true;
  }
  return false;
}

fn CalcVolumeForceForElems(domain: *Domain) void
{
   const numElem = domain.numElem;
   if (numElem != 0) {
      const hgcoef = domain.hgcoef;
      const sigxx  = domain.sigxx;
      const sigyy  = domain.sigyy;
      const sigzz  = domain.sigzz;
      const determ = domain.determ;

      // Sum contributions to total stress tensor
      InitStressTermsForElems(domain.p, domain.q,
                              sigxx, sigyy, sigzz, numElem);

      // call elemlib stress integration loop to produce nodal forces from
      // material stresses.
      IntegrateStressForElems( domain.nodelist,
                               domain.x, domain.y, domain.z,
                               domain.fx, domain.fy, domain.fz,
                               sigxx, sigyy, sigzz, determ, numElem);

      // check for negative element volume
      if (VolErr1(determ, numElem))
         std.process.exit(@intFromEnum(Err.VolumeError));

      CalcHourglassControlForElems(domain, determ, hgcoef);
   }
}

fn CalcForceForNodes(domain: *Domain) void
{
  const numNode = domain.numNode;
  var fx = domain.fx;
  var fy = domain.fy;
  var fz = domain.fz;

  var i: Index_t = 0;
  while (i < numNode) : ( i += 1 ) {
     fx[i] = ZERO;
     fy[i] = ZERO;
     fz[i] = ZERO;
  }
}

fn CalcAccelerationForNodes(xdd: [*]Real_t, ydd: [*]Real_t, zdd: [*]Real_t,
                            fx: [*]Real_t, fy: [*]Real_t, fz: [*]Real_t,
                            nodalMass: [*]Real_t, numNode: Index_t) void
{
   var k: Index_t = 0;
   while ( k < numNode ) : ( k += 1 ) {
      xdd[k] = fx[k] / nodalMass[k];
      ydd[k] = fy[k] / nodalMass[k];
      zdd[k] = fz[k] / nodalMass[k];
   }
}

fn ApplyAccelerationBoundaryConditionsForNodes(xdd: [*]Real_t, ydd: [*]Real_t,
                                               zdd: [*]Real_t,
                                               symmX: [*]Index_t,
                                               symmY: [*]Index_t,
                                               symmZ: [*]Index_t,
                                               size: Index_t) void
{
  const numNodeBC = (size+1)*(size+1);

  var k: Index_t = 0;
  while ( k < numNodeBC ) : ( k += 1 ) {
     xdd[symmX[k]] = ZERO;
     ydd[symmY[k]] = ZERO;
     zdd[symmZ[k]] = ZERO;
  }
}

fn CalcVelocityForNodes(xd: [*]Real_t,  yd: [*]Real_t,  zd: [*]Real_t,
                        xdd: [*]Real_t, ydd: [*]Real_t, zdd: [*]Real_t,
                        dt: Real_t, u_cut: Real_t, numNode: Index_t)
                        void
{
  var idx: Index_t = 0;
  while ( idx < numNode ) : ( idx += 1 ) {

     const xt = xd[idx] + xdd[idx] * dt;
     xd[idx] = if ( @abs(xt) < u_cut ) ZERO else xt;

     const yt = yd[idx] + ydd[idx] * dt;
     yd[idx] = if( @abs(yt) < u_cut ) ZERO else yt;

     const zt = zd[idx] + zdd[idx] * dt;
     zd[idx] = if( @abs(zt) < u_cut ) ZERO else zt;
   }
}

fn CalcPositionForNodes(x: [*]Real_t, y: [*]Real_t, z: [*]Real_t,
                        xd: [*]Real_t, yd: [*]Real_t, zd: [*]Real_t,
                        dt: Real_t, numNode: Index_t) void
{
   var k: Index_t = 0;
   while ( k < numNode ) : ( k += 1 ) {
     x[k] += xd[k] * dt;
     y[k] += yd[k] * dt;
     z[k] += zd[k] * dt;
   }
}

fn LagrangeNodal(domain: *Domain) void
{
  const  delt = domain.deltatime;
  const u_cut = domain.u_cut;

  // time of boundary condition evaluation is beginning of step for force and
  // acceleration boundary conditions.
  CalcForceForNodes(domain);

  // Calcforce calls partial, force, hourq
  CalcVolumeForceForElems(domain);

  // Calculate Nodal Forces at domain boundaries
  // problem->commSBN->Transfer(CommSBN::forces);

  CalcAccelerationForNodes(domain.xdd, domain.ydd, domain.zdd,
                           domain.fx, domain.fy, domain.fz,
                           domain.nodalMass, domain.numNode);

  ApplyAccelerationBoundaryConditionsForNodes(domain.xdd, domain.ydd,
                                              domain.zdd, domain.symmX,
                                              domain.symmY, domain.symmZ,
                                              domain.sizeX);

  CalcVelocityForNodes(domain.xd,  domain.yd,  domain.zd,
                       domain.xdd, domain.ydd, domain.zdd,
                       delt, u_cut, domain.numNode);

  CalcPositionForNodes(domain.x,  domain.y,  domain.z,
                       domain.xd, domain.yd, domain.zd,
                       delt, domain.numNode );

  return;
}

fn TRIPLE_PRODUCT(x1_: Real_t, y1_: Real_t, z1_: Real_t,
                  x2_: Real_t, y2_: Real_t, z2_: Real_t,
                  x3_: Real_t, y3_: Real_t, z3_: Real_t) Real_t
{
   return  (x1_*(y2_*z3_ - z2_*y3_) +
            x2_*(z1_*y3_ - y1_*z3_) +
            x3_*(y1_*z2_ - z1_*y2_));
}


fn CalcElemVolume2(x0: Real_t, x1: Real_t, x2: Real_t, x3: Real_t,
                   x4: Real_t, x5: Real_t, x6: Real_t, x7: Real_t,
                   y0: Real_t, y1: Real_t, y2: Real_t, y3: Real_t,
                   y4: Real_t, y5: Real_t, y6: Real_t, y7: Real_t,
                   z0: Real_t, z1: Real_t, z2: Real_t, z3: Real_t,
                   z4: Real_t, z5: Real_t, z6: Real_t, z7: Real_t) Real_t
{
  var fv: Real_t = ZERO;
  {
     const dx31 = x3 - x1;
     const dy31 = y3 - y1;
     const dz31 = z3 - z1;

     const dx72 = x7 - x2;
     const dy72 = y7 - y2;
     const dz72 = z7 - z2;

     const s1 = dx31 + dx72;
     const s2 = dy31 + dy72;
     const s3 = dz31 + dz72;

     const dx63 = x6 - x3;
     const dy63 = y6 - y3;
     const dz63 = z6 - z3;

     const dx20 = x2 - x0;
     const dy20 = y2 - y0;
     const dz20 = z2 - z0;

     fv += TRIPLE_PRODUCT(s1, dx63, dx20,
                          s2, dy63, dy20,
                          s3, dz63, dz20);
  }
  {
     const dx43 = x4 - x3;
     const dy43 = y4 - y3;
     const dz43 = z4 - z3;

     const dx57 = x5 - x7;
     const dy57 = y5 - y7;
     const dz57 = z5 - z7;

     const s1 = dx43 + dx57;
     const s2 = dy43 + dy57;
     const s3 = dz43 + dz57;

     const dx64 = x6 - x4;
     const dy64 = y6 - y4;
     const dz64 = z6 - z4;

     const dx70 = x7 - x0;
     const dy70 = y7 - y0;
     const dz70 = z7 - z0;

     fv += TRIPLE_PRODUCT(s1, dx64, dx70,
                          s2, dy64, dy70,
                          s3, dz64, dz70);
  }
  {
     const dx14 = x1 - x4;
     const dy14 = y1 - y4;
     const dz14 = z1 - z4;

     const dx25 = x2 - x5;
     const dy25 = y2 - y5;
     const dz25 = z2 - z5;

     const s1 = dx14 + dx25;
     const s2 = dy14 + dy25;
     const s3 = dz14 + dz25;

     const dx61 = x6 - x1;
     const dy61 = y6 - y1;
     const dz61 = z6 - z1;

     const dx50 = x5 - x0;
     const dy50 = y5 - y0;
     const dz50 = z5 - z0;

     fv += TRIPLE_PRODUCT(s1, dx61, dx50,
                          s2, dy61, dy50,
                          s3, dz61, dz50);
   }

   return (fv  / TWELVE);
}

fn CalcElemVolume(x: [8]Real_t, y: [8]Real_t, z: [8]Real_t) Real_t
{
   return
   CalcElemVolume2( x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
                    y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7],
                    z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);
}

fn AreaFace(x0: Real_t, x1: Real_t, x2: Real_t, x3: Real_t,
            y0: Real_t, y1: Real_t, y2: Real_t, y3: Real_t,
            z0: Real_t, z1: Real_t, z2: Real_t, z3: Real_t) Real_t
{
   const dx1 = (x2 - x0);
   const dx2 = (x3 - x1);
   const dy1 = (y2 - y0);
   const dy2 = (y3 - y1);
   const dz1 = (z2 - z0);
   const dz2 = (z3 - z1);
   const fx = dx1 - dx2;
   const fy = dy1 - dy2;
   const fz = dz1 - dz2;
   const gx = dx1 + dx2;
   const gy = dy1 + dy2;
   const gz = dz1 + dz2;
   const term1 = fx * fx + fy * fy + fz * fz;
   const term2 = gx * gx + gy * gy + gz * gz;
   const term3 = fx * gx + fy * gy + fz * gz;
   return  term1 * term2 + term3 * term3 ;
}

fn CalcElemCharacteristicLength(x: [8]Real_t, y: [8]Real_t, z: [8]Real_t,
                                volume: Real_t) Real_t
{
   var charLength: Real_t = ZERO;
   var a: Real_t =
       AreaFace(x[0],x[1],x[2],x[3],
                y[0],y[1],y[2],y[3],
                z[0],z[1],z[2],z[3]);
   if (a > charLength) charLength = a;

   a = AreaFace(x[4],x[5],x[6],x[7],
                y[4],y[5],y[6],y[7],
                z[4],z[5],z[6],z[7]);
   if (a > charLength) charLength = a;

   a = AreaFace(x[0],x[1],x[5],x[4],
                y[0],y[1],y[5],y[4],
                z[0],z[1],z[5],z[4]);
   if (a > charLength) charLength = a;

   a = AreaFace(x[1],x[2],x[6],x[5],
                y[1],y[2],y[6],y[5],
                z[1],z[2],z[6],z[5]);
   if (a > charLength) charLength = a;

   a = AreaFace(x[2],x[3],x[7],x[6],
                y[2],y[3],y[7],y[6],
                z[2],z[3],z[7],z[6]);
   if (a > charLength) charLength = a;

   a = AreaFace(x[3],x[0],x[4],x[7],
                y[3],y[0],y[4],y[7],
                z[3],z[0],z[4],z[7]);
   if (a > charLength) charLength = a;

   return (4.0 * volume) / math.sqrt( charLength );
}

fn CalcElemVelocityGradient(xvel: [8]Real_t, yvel: [8]Real_t,
                            zvel: [8]Real_t,
                            b: [3][8]Real_t, detJ: Real_t, d: []Real_t)
                            void
{
   const inv_detJ = ONE / detJ;
   const pfx = b[0];
   const pfy = b[1];
   const pfz = b[2];

   const  dxv0 = (xvel[0]-xvel[6]);
   const  dxv1 = (xvel[1]-xvel[7]);
   const  dxv2 = (xvel[2]-xvel[4]);
   const  dxv3 = (xvel[3]-xvel[5]);
   const  dyv0 = (yvel[0]-yvel[6]);
   const  dyv1 = (yvel[1]-yvel[7]);
   const  dyv2 = (yvel[2]-yvel[4]);
   const  dyv3 = (yvel[3]-yvel[5]);
   const  dzv0 = (zvel[0]-zvel[6]);
   const  dzv1 = (zvel[1]-zvel[7]);
   const  dzv2 = (zvel[2]-zvel[4]);
   const  dzv3 = (zvel[3]-zvel[5]);

   d[0] = inv_detJ * ( pfx[0] * dxv0
                     + pfx[1] * dxv1
                     + pfx[2] * dxv2
                     + pfx[3] * dxv3 );

   d[1] = inv_detJ * ( pfy[0] * dyv0
                     + pfy[1] * dyv1
                     + pfy[2] * dyv2
                     + pfy[3] * dyv3 );

   d[2] = inv_detJ * ( pfz[0] * dzv0
                     + pfz[1] * dzv1
                     + pfz[2] * dzv2
                     + pfz[3] * dzv3 );

   const dyddx = inv_detJ *
      ( pfx[0] * dyv0 + pfx[1] * dyv1
      + pfx[2] * dyv2 + pfx[3] * dyv3 );

   const dxddy = inv_detJ *
      ( pfy[0] * dxv0 + pfy[1] * dxv1
      + pfy[2] * dxv2 + pfy[3] * dxv3 );

   const dzddx  = inv_detJ *
      ( pfx[0] * dzv0 + pfx[1] * dzv1
      + pfx[2] * dzv2 + pfx[3] * dzv3 );

   const dxddz  = inv_detJ *
      ( pfz[0] * dxv0 + pfz[1] * dxv1
      + pfz[2] * dxv2 + pfz[3] * dxv3 );

   const dzddy  = inv_detJ *
      ( pfy[0] * dzv0 + pfy[1] * dzv1
      + pfy[2] * dzv2 + pfy[3] * dzv3 );

   const dyddz  = inv_detJ *
      ( pfz[0] * dyv0 + pfz[1] * dyv1
      + pfz[2] * dyv2 + pfz[3] * dyv3 );

   d[5]  = ( dxddy + dyddx ) * HALF;
   d[4]  = ( dxddz + dzddx ) * HALF;
   d[3]  = ( dzddy + dyddz ) * HALF;
}

fn UpdatePos(deltaTime: Real_t,
             x_local: []Real_t, y_local: []Real_t, z_local: []Real_t,
             xd_local: [8]Real_t, yd_local: [8]Real_t, zd_local: [8]Real_t)
             void
{
  const dt2 = deltaTime * HALF;
  var idx: Index_t = 0;
  while ( idx < 8 ) : ( idx += 1 ) {
    x_local[idx] -= dt2 * xd_local[idx];
    y_local[idx] -= dt2 * yd_local[idx];
    z_local[idx] -= dt2 * zd_local[idx];
  }
}

fn CalcKinematicsForElems(nodelist: [*][8]Index_t,
                          x: [*]Real_t, y: [*]Real_t, z: [*]Real_t,
                          xd: [*]Real_t, yd: [*]Real_t, zd: [*]Real_t,
                          dxx: [*]Real_t, dyy: [*]Real_t, dzz: [*]Real_t,
                          v: [*]Real_t, volo: [*]Real_t,
                          vnew: [*]Real_t, delv: [*]Real_t, arealg: [*]Real_t,
                          deltaTime: Real_t, numElem: Index_t) void
{
  // loop over all elements
  var k: Index_t = 0;
  while ( k < numElem ) : ( k += 1 ) {
    const elemToNode = nodelist[k];

    var  x_local: [8]Real_t = undefined;
    var  y_local: [8]Real_t = undefined;
    var  z_local: [8]Real_t = undefined;
    var xd_local: [8]Real_t = undefined;
    var yd_local: [8]Real_t = undefined;
    var zd_local: [8]Real_t = undefined;

    // shape function derivatives
    var D: [6]Real_t = undefined;
    var B: [3][8]Real_t = undefined;

    // get nodal coordinates from global arrays and copy into local arrays.
    GatherNodes(elemToNode, x, y, z, &x_local, &y_local, &z_local);

    // volume calculations
    const volume = CalcElemVolume(x_local, y_local, z_local);
    const relativeVolume = volume / volo[k];
    vnew[k] = relativeVolume;
    delv[k] = relativeVolume - v[k];

    // set characteristic length
    arealg[k] = CalcElemCharacteristicLength(x_local, y_local, z_local,
                                             volume);

    // get nodal velocities from global array and copy into local arrays.
    GatherNodes(elemToNode, xd, yd, zd, &xd_local, &yd_local, &zd_local);

    UpdatePos(deltaTime, &x_local, &y_local, &z_local,
              xd_local, yd_local, zd_local);

    const detJ =
       CalcElemShapeFunctionDerivatives(x_local, y_local, z_local, &B);

    CalcElemVelocityGradient( xd_local, yd_local, zd_local,
                               B, detJ, &D );

    // put velocity gradient quantities into their global arrays.
    dxx[k] = D[0];
    dyy[k] = D[1];
    dzz[k] = D[2];
  }
}

fn VolErr2(domain: *Domain, numElem: Index_t) bool
{
  var vdovv = domain.vdov;
  var dxx = domain.dxx;
  var dyy = domain.dyy;
  var dzz = domain.dzz;
  const vnew = domain.vnew;

  var k: Index_t = 0;
  while ( k < numElem ) : ( k += 1 ) {
    // calc strain rate and apply as constraint (only done in FB element)
    const vdov = dxx[k] + dyy[k] + dzz[k];
    const vdovthird = vdov * (ONE / THREE );

    // make the rate of deformation tensor deviatoric
    vdovv[k] = vdov;
    dxx[k] -= vdovthird;
    dyy[k] -= vdovthird;
    dzz[k] -= vdovthird;

    // See if any volumes are negative, and take appropriate action.
    if (vnew[k] <= ZERO)
    {
      return true;
    }
  }
  return false;
}

fn CalcLagrangeElements(domain: *Domain) void
{
   const numElem = domain.numElem;
   if (numElem > 0) {
      const deltatime = domain.deltatime;

      CalcKinematicsForElems(domain.nodelist,
                             domain.x, domain.y, domain.z,
                             domain.xd, domain.yd, domain.zd,
                             domain.dxx, domain.dyy, domain.dzz,
                             domain.v, domain.volo,
                             domain.vnew, domain.delv, domain.arealg,
                             deltatime, numElem);

      // element loop to do some stuff not included in the elemlib function.
      if (VolErr2(domain, numElem))
         std.process.exit(@intFromEnum(Err.VolumeError));
   }
}

fn CalcMonotonicQGradientsForElems(x: [*]Real_t, y: [*]Real_t, z: [*]Real_t,
                                   xd: [*]Real_t, yd: [*]Real_t, zd: [*]Real_t,
                                   volo: [*]Real_t, vnew: [*]Real_t,
                                   delv_xi: [*]Real_t,
                                   delv_eta: [*]Real_t,
                                   delv_zeta: [*]Real_t,
                                   delx_xi: [*]Real_t,
                                   delx_eta: [*]Real_t,
                                   delx_zeta: [*]Real_t,
                                   nodelist: [*][8]Index_t,
                                   numElem: Index_t) void
{
   var idx: Index_t = 0;
   while ( idx < numElem ) : ( idx += 1 ) {
      const ptiny = 1.0e-36;

      const elemToNode = nodelist[idx];

      const n0 = elemToNode[0];
      const n1 = elemToNode[1];
      const n2 = elemToNode[2];
      const n3 = elemToNode[3];
      const n4 = elemToNode[4];
      const n5 = elemToNode[5];
      const n6 = elemToNode[6];
      const n7 = elemToNode[7];

      const vol = volo[idx]*vnew[idx];
      const norm = ONE / ( vol + ptiny );

      const x0 = x[n0];
      const x1 = x[n1];
      const x2 = x[n2];
      const x3 = x[n3];
      const x4 = x[n4];
      const x5 = x[n5];
      const x6 = x[n6];
      const x7 = x[n7];

      const y0 = y[n0];
      const y1 = y[n1];
      const y2 = y[n2];
      const y3 = y[n3];
      const y4 = y[n4];
      const y5 = y[n5];
      const y6 = y[n6];
      const y7 = y[n7];

      const z0 = z[n0];
      const z1 = z[n1];
      const z2 = z[n2];
      const z3 = z[n3];
      const z4 = z[n4];
      const z5 = z[n5];
      const z6 = z[n6];
      const z7 = z[n7];

      const dxj = -QUARTER*((x0 + x1 + x5 + x4) - (x3 + x2 + x6 + x7));
      const dyj = -QUARTER*((y0 + y1 + y5 + y4) - (y3 + y2 + y6 + y7));
      const dzj = -QUARTER*((z0 + z1 + z5 + z4) - (z3 + z2 + z6 + z7));

      const dxi = QUARTER*((x1 + x2 + x6 + x5) - (x0 + x3 + x7 + x4));
      const dyi = QUARTER*((y1 + y2 + y6 + y5) - (y0 + y3 + y7 + y4));
      const dzi = QUARTER*((z1 + z2 + z6 + z5) - (z0 + z3 + z7 + z4));

      const dxk = QUARTER*((x4 + x5 + x6 + x7) - (x0 + x1 + x2 + x3));
      const dyk = QUARTER*((y4 + y5 + y6 + y7) - (y0 + y1 + y2 + y3));
      const dzk = QUARTER*((z4 + z5 + z6 + z7) - (z0 + z1 + z2 + z3));

      // find delvk and delxk ( i cross j )

      var ax: Real_t = dyi*dzj - dzi*dyj;
      var ay: Real_t = dzi*dxj - dxi*dzj;
      var az: Real_t = dxi*dyj - dyi*dxj;

      // i type and rhs type conflict?
      delx_zeta[idx] = ( vol / math.sqrt( ax*ax + ay*ay + az*az + ptiny ) );

      const xv0 = xd[n0];
      const xv1 = xd[n1];
      const xv2 = xd[n2];
      const xv3 = xd[n3];
      const xv4 = xd[n4];
      const xv5 = xd[n5];
      const xv6 = xd[n6];
      const xv7 = xd[n7];

      const yv0 = yd[n0];
      const yv1 = yd[n1];
      const yv2 = yd[n2];
      const yv3 = yd[n3];
      const yv4 = yd[n4];
      const yv5 = yd[n5];
      const yv6 = yd[n6];
      const yv7 = yd[n7];

      const zv0 = zd[n0];
      const zv1 = zd[n1];
      const zv2 = zd[n2];
      const zv3 = zd[n3];
      const zv4 = zd[n4];
      const zv5 = zd[n5];
      const zv6 = zd[n6];
      const zv7 = zd[n7];

      var dxv: Real_t =
                QUARTER*((xv4 + xv5 + xv6 + xv7) - (xv0 + xv1 + xv2 + xv3));
      var dyv: Real_t =
                QUARTER*((yv4 + yv5 + yv6 + yv7) - (yv0 + yv1 + yv2 + yv3));
      var dzv: Real_t =
                QUARTER*((zv4 + zv5 + zv6 + zv7) - (zv0 + zv1 + zv2 + zv3));

      delv_zeta[idx] = ( ax*dxv + ay*dyv + az*dzv ) * norm;

      // find delxi and delvi ( j cross k )

      ax = dyj*dzk - dzj*dyk;
      ay = dzj*dxk - dxj*dzk;
      az = dxj*dyk - dyj*dxk;

      delx_xi[idx] = vol / math.sqrt(ax*ax + ay*ay + az*az + ptiny);

      dxv = QUARTER*((xv1 + xv2 + xv6 + xv5) - (xv0 + xv3 + xv7 + xv4));
      dyv = QUARTER*((yv1 + yv2 + yv6 + yv5) - (yv0 + yv3 + yv7 + yv4));
      dzv = QUARTER*((zv1 + zv2 + zv6 + zv5) - (zv0 + zv3 + zv7 + zv4));

      delv_xi[idx] = ( ax*dxv + ay*dyv + az*dzv ) * norm;

      // find delxj and delvj ( k cross i )

      ax = dyk*dzi - dzk*dyi;
      ay = dzk*dxi - dxk*dzi;
      az = dxk*dyi - dyk*dxi;

      delx_eta[idx] = vol / math.sqrt(ax*ax + ay*ay + az*az + ptiny);

      dxv = -QUARTER*((xv0 + xv1 + xv5 + xv4) - (xv3 + xv2 + xv6 + xv7));
      dyv = -QUARTER*((yv0 + yv1 + yv5 + yv4) - (yv3 + yv2 + yv6 + yv7));
      dzv = -QUARTER*((zv0 + zv1 + zv5 + zv4) - (zv3 + zv2 + zv6 + zv7));

      // i is int type and ax is Real_t == mismatch
      delv_eta[idx] = ( ax*dxv + ay*dyv + az*dzv ) * norm;
   }
}

fn CalcMonotonicQRegionForElems(elemBC: [*]u32,
                                lxim: [*]Index_t, lxip: [*]Index_t,
                                letam: [*]Index_t, letap: [*]Index_t,
                                lzetam: [*]Index_t, lzetap: [*]Index_t,
                                delv_xi: [*]Real_t, delv_eta: [*]Real_t,
                                delv_zeta: [*]Real_t, delx_xi: [*]Real_t,
                                delx_eta: [*]Real_t, delx_zeta: [*]Real_t,
                                vdov: [*]Real_t, volo: [*]Real_t,
                                vnew: [*]Real_t, elemMass: [*]Real_t,
                                qq: [*]Real_t, ql: [*]Real_t,
                                qlc_monoq: Real_t, qqc_monoq: Real_t,
                                monoq_limiter_mult: Real_t,
                                monoq_max_slope: Real_t,
                                ptiny: Real_t, numElem: Index_t) void
{
   var idx: Index_t = 0;
   while ( idx < numElem ) : ( idx += 1 ) {
      const bcMask: u32 = elemBC[idx];

      //  phixi
      var norm: Real_t = ONE / ( delv_xi[idx] + ptiny );

      var delvm: Real_t = switch (bcMask & BCbits(BC.XI_M)) {
         0                    => delv_xi[lxim[idx]],
         BCbits(BC.XI_M_SYMM) => delv_xi[idx],
         BCbits(BC.XI_M_FREE) => ZERO,
         else                 => undefined,
      };
      var delvp : Real_t = switch (bcMask & BCbits(BC.XI_P)) {
         0                    => delv_xi[lxip[idx]],
         BCbits(BC.XI_P_SYMM) => delv_xi[idx],
         BCbits(BC.XI_P_FREE) => ZERO,
         else                 => undefined,
      };

      delvm = delvm * norm;
      delvp = delvp * norm;

      var phixi: Real_t = ( delvm + delvp ) * HALF;

      delvm *= monoq_limiter_mult;
      delvp *= monoq_limiter_mult;

      if ( delvm < phixi ) phixi = delvm;
      if ( delvp < phixi ) phixi = delvp;
      if ( phixi < ZERO) phixi = ZERO;
      if ( phixi > monoq_max_slope) phixi = monoq_max_slope;


      //  phieta
      norm = ONE / ( delv_eta[idx] + ptiny );

      delvm = switch (bcMask & BCbits(BC.ETA_M)) {
         0                      => delv_eta[letam[idx]],
         BCbits(BC.ETA_M_SYMM)  => delv_eta[idx],
         BCbits(BC.ETA_M_FREE)  => ZERO,
         else                   => undefined,
      };
      delvp = switch (bcMask & BCbits(BC.ETA_P)) {
         0                      => delv_eta[letap[idx]],
         BCbits(BC.ETA_P_SYMM)  => delv_eta[idx],
         BCbits(BC.ETA_P_FREE)  => ZERO,
         else                   => undefined,
      };

      delvm = delvm * norm;
      delvp = delvp * norm;

      var phieta: Real_t = ( delvm + delvp ) * HALF;

      delvm *= monoq_limiter_mult;
      delvp *= monoq_limiter_mult;

      if ( delvm  < phieta ) phieta = delvm;
      if ( delvp  < phieta ) phieta = delvp;
      if ( phieta < ZERO) phieta = ZERO;
      if ( phieta > monoq_max_slope)  phieta = monoq_max_slope;

      //  phizeta
      norm = ONE / ( delv_zeta[idx] + ptiny );

      delvm = switch (bcMask & BCbits(BC.ZETA_M)) {
         0                      => delv_zeta[lzetam[idx]],
         BCbits(BC.ZETA_M_SYMM) => delv_zeta[idx],
         BCbits(BC.ZETA_M_FREE) => ZERO,
         else                   => undefined,
      };
      delvp = switch (bcMask & BCbits(BC.ZETA_P)) {
         0                      => delv_zeta[lzetap[idx]],
         BCbits(BC.ZETA_P_SYMM) => delv_zeta[idx],
         BCbits(BC.ZETA_P_FREE) => ZERO,
         else                   => undefined,
      };

      delvm = delvm * norm;
      delvp = delvp * norm;

      var phizeta: Real_t = ( delvm + delvp ) * HALF;

      delvm *= monoq_limiter_mult;
      delvp *= monoq_limiter_mult;

      if ( delvm   < phizeta ) phizeta = delvm;
      if ( delvp   < phizeta ) phizeta = delvp;
      if ( phizeta < ZERO)     phizeta = ZERO;
      if ( phizeta > monoq_max_slope  ) phizeta = monoq_max_slope;

      // Remove length scale

      var qlin: Real_t = undefined;
      var qquad: Real_t = undefined;

      if ( vdov[idx] > ZERO )  {
         qlin  = ZERO;
         qquad = ZERO;
      }
      else {
         var delvxxi: Real_t   = delv_xi[idx]   * delx_xi[idx];
         var delvxeta: Real_t  = delv_eta[idx]  * delx_eta[idx];
         var delvxzeta: Real_t = delv_zeta[idx] * delx_zeta[idx];

         if ( delvxxi   > ZERO ) delvxxi   = ZERO;
         if ( delvxeta  > ZERO ) delvxeta  = ZERO;
         if ( delvxzeta > ZERO ) delvxzeta = ZERO;

         const rho = elemMass[idx] / (volo[idx] * vnew[idx]);

         qlin = -qlc_monoq * rho *
            (  delvxxi   * (ONE - phixi) +
               delvxeta  * (ONE - phieta) +
               delvxzeta * (ONE - phizeta)  );

         qquad = qqc_monoq * rho *
            (  delvxxi*delvxxi     * (ONE - phixi*phixi) +
               delvxeta*delvxeta   * (ONE - phieta*phieta) +
               delvxzeta*delvxzeta * (ONE - phizeta*phizeta)  );
      }

      qq[idx] = qquad;
      ql[idx] = qlin;
   }
}

fn CalcMonotonicQForElems(domain: *Domain) void
{  
   //
   // calculate the monotonic q for pure regions
   //
   const numElem = domain.numElem;
   if (numElem > 0) {
      //
      // initialize parameters
      // 
      const ptiny = 1.e-36;

      CalcMonotonicQRegionForElems(
                           domain.elemBC,
                           domain.lxim,   domain.lxip,
                           domain.letam,  domain.letap,
                           domain.lzetam, domain.lzetap,
                           domain.delv_xi, domain.delv_eta, domain.delv_zeta,
                           domain.delx_xi, domain.delx_eta, domain.delx_zeta,
                           domain.vdov, domain.volo, domain.vnew,
                           domain.elemMass, domain.qq, domain.ql,
                           domain.qlc_monoq, domain.qqc_monoq,
                           domain.monoq_limiter_mult,
                           domain.monoq_max_slope,
                           ptiny, numElem );
   }
}

fn Qerr(q: [*]Real_t, numElem: Index_t, qstop:Real_t) Index_t
{
  var idx: Index_t = math.maxInt(Index_t);
  var k: Index_t = 0;
  while ( k < numElem ) : ( k += 1 ) {
    if ( q[k] > qstop ) {
      idx = k;
      // break;
    }
  }
  return idx;
}

fn CalcQForElems(domain: *Domain) void
{
   //
   // MONOTONIC Q option
   //

   const numElem = domain.numElem;

   if (numElem != 0) {
      // Calculate velocity gradients, applied at the domain level
      CalcMonotonicQGradientsForElems(domain.x,  domain.y,  domain.z,
                                      domain.xd, domain.yd, domain.zd,
                                      domain.volo, domain.vnew,
                                      domain.delv_xi,
                                      domain.delv_eta,
                                      domain.delv_zeta,
                                      domain.delx_xi,
                                      domain.delx_eta,
                                      domain.delx_zeta,
                                      domain.nodelist,
                                      numElem);

      // This will be applied at the region level
      CalcMonotonicQForElems(domain);

      // Don't allow excessive artificial viscosity
      if (Qerr(domain.q, numElem, domain.qstop) != math.maxInt(Index_t))
         std.process.exit(@intFromEnum(Err.QStopError));
   }
}


fn CalcPressureForElems(p_new: [*]Real_t, bvc: [*]Real_t,
                        pbvc: [*]Real_t, e_old: [*]Real_t,
                        compression: [*]Real_t, vnewc: [*]Real_t,
                        pmin: Real_t, p_cut: Real_t, eosvmax: Real_t,
                        length:Index_t) void
{
   const c1s = TWO / THREE ;

   var idx: Index_t = 0;
   while( idx < length ) : ( idx += 1 ) {
      bvc[idx] = c1s * (compression[idx] + ONE );
      pbvc[idx] = c1s;
   }

   idx = 0;
   while( idx < length ) : ( idx += 1 ) {
      p_new[idx] = bvc[idx] * e_old[idx];

      if (@abs(p_new[idx]) <  p_cut)
         p_new[idx] = ZERO;

      if ( vnewc[idx] >= eosvmax ) // impossible condition here?
         p_new[idx] = ZERO;

      if (p_new[idx] < pmin)
         p_new[idx] = pmin;
   }
}

fn CalcEnergyForElems(p_new: [*]Real_t, e_new: [*]Real_t, q_new: [*]Real_t,
                      bvc: [*]Real_t, pbvc: [*]Real_t,
                      p_old: [*]Real_t, e_old: [*]Real_t, q_old: [*]Real_t,
                      compression: [*]Real_t, compHalfStep: [*]Real_t,
                      vnewc: [*]Real_t, work: [*]Real_t, delvc: [*]Real_t,
                      pmin: Real_t, p_cut: Real_t,  e_cut: Real_t,
                      q_cut: Real_t, emin: Real_t, qq_old: [*]Real_t,
                      ql_old: [*]Real_t, rho0: Real_t, eosvmax: Real_t,
                      pHalfStep: [*]Real_t, length: Index_t) void
{
   const sixth = ONE / SIX;

   var idx: Index_t = 0;
   while( idx < length ) : ( idx += 1 ) {
      e_new[idx] = e_old[idx] - delvc[idx]*(p_old[idx] + q_old[idx]) * HALF +
                 work[idx] * HALF;

      if (e_new[idx]  < emin ) {
         e_new[idx] = emin;
      }
   }

   CalcPressureForElems(pHalfStep, bvc, pbvc, e_new, compHalfStep, vnewc,
                   pmin, p_cut, eosvmax, length);

   idx = 0;
   while( idx < length ) : ( idx += 1 ) {
      const vhalf = ONE / (ONE + compHalfStep[idx]);

      if ( delvc[idx] > ZERO ) {
         q_new[idx] = ZERO;
      }
      else {
         var ssc: Real_t = ( pbvc[idx] * e_new[idx]
                 + vhalf * vhalf * bvc[idx] * pHalfStep[idx] ) / rho0;

         if ( ssc > 0.1111111e-36 ) {
            ssc = math.sqrt(ssc);
         } else {
            ssc = 0.3333333e-18;
         }

         q_new[idx] = (ssc*ql_old[idx] + qq_old[idx]);
      }

      e_new[idx] = e_new[idx] + delvc[idx] * HALF
         * (  THREE * (p_old[idx]     + q_old[idx])
            - FOUR * (pHalfStep[idx] + q_new[idx]));
   }

   idx = 0;
   while( idx < length ) : ( idx += 1 ) {

      e_new[idx] += work[idx] * HALF;

      if (@abs(e_new[idx]) < e_cut) {
         e_new[idx] = ZERO;
      }
      if (e_new[idx] < emin ) {
         e_new[idx] = emin;
      }
   }

   CalcPressureForElems(p_new, bvc, pbvc, e_new, compression, vnewc,
                   pmin, p_cut, eosvmax, length);

   idx = 0;
   while( idx < length ) : ( idx += 1 ) {
      var q_tilde: Real_t = undefined;

      if (delvc[idx] > ZERO) {
         q_tilde = ZERO;
      }
      else {
         var ssc: Real_t = ( pbvc[idx] * e_new[idx]
                 + vnewc[idx] * vnewc[idx] * bvc[idx] * p_new[idx] ) / rho0;

         if ( ssc > 0.1111111e-36 ) {
            ssc = math.sqrt(ssc);
         } else {
            ssc = 0.3333333e-18;
         }

         q_tilde = (ssc*ql_old[idx] + qq_old[idx]);
      }

      e_new[idx] = e_new[idx] - ( (p_old[idx] + q_old[idx]) * SEVEN
                                - (pHalfStep[idx] + q_new[idx]) * EIGHT
                                + (p_new[idx] + q_tilde)) * delvc[idx]*sixth;

      if (@abs(e_new[idx]) < e_cut) {
         e_new[idx] = ZERO;
      }
      if ( e_new[idx] < emin ) {
         e_new[idx] = emin;
      }
   }

   CalcPressureForElems(p_new, bvc, pbvc, e_new, compression, vnewc,
                   pmin, p_cut, eosvmax, length);

   idx = 0;
   while( idx < length ) : ( idx += 1 ) {

      if ( delvc[idx] <= ZERO ) {
         var ssc: Real_t = ( pbvc[idx] * e_new[idx]
                 + vnewc[idx] * vnewc[idx] * bvc[idx] * p_new[idx] ) / rho0;

         if ( ssc > 0.1111111e-36 ) {
            ssc = math.sqrt(ssc);
         } else {
            ssc = 0.3333333e-18;
         }

         q_new[idx] = (ssc*ql_old[idx] + qq_old[idx]);

         if (@abs(q_new[idx]) < q_cut) q_new[idx] = ZERO;
      }
   }

   return ;
}

fn CalcSoundSpeedForElems(length: Index_t, ss: [*]Real_t,
                          vnewc: [*]Real_t, rho0: Real_t, enewc: [*]Real_t,
                          pnewc: [*]Real_t, pbvc: [*]Real_t,
                          bvc: [*]Real_t) void
{
   var idx: Index_t = 0;
   while( idx < length ) : ( idx += 1 ) {
      var ssTmp: Real_t = (pbvc[idx] * enewc[idx] + vnewc[idx] * vnewc[idx] *
                 bvc[idx] * pnewc[idx]) / rho0;
      if (ssTmp <= 0.1111111e-36) {
         ssTmp = 0.3333333e-18;
      }
      else {
         ssTmp = math.sqrt( ssTmp );
      }
      ss[idx] = ssTmp;
   }
}

fn EvalCopy(p_old: [*]Real_t, p: [*]Real_t, numElem: Index_t) void
{
  var idx: Index_t = 0;
  while( idx < numElem ) : ( idx += 1 ) {
    p_old[idx] = p[idx];
  }
}

fn EvalCompression(compression: [*]Real_t, compHalfStep: [*]Real_t,
                   numElem: Index_t, vnewc: [*]Real_t, delvc: [*]Real_t) void
{
  var idx: Index_t = 0;
  while( idx < numElem ) : ( idx += 1 ) {
    compression[idx] = ONE / vnewc[idx] - ONE;
    const vchalf = vnewc[idx] - delvc[idx] * HALF;
    compHalfStep[idx] = ONE / vchalf - ONE;
  }
}

fn EvalEosVmin(vnewc: [*]Real_t, compHalfStep: [*]Real_t,
               compression: [*]Real_t,
               numElem: Index_t, eosvmin: Real_t) void
{
  var idx: Index_t = 0;
  while( idx < numElem ) : ( idx += 1 ) {
    if (vnewc[idx] <= eosvmin) { // impossible due to calling func?
      compHalfStep[idx] = compression[idx];
    }
  }
}

fn EvalEosVmax(vnewc: [*]Real_t, p_old: [*]Real_t,
               compHalfStep: [*]Real_t, compression: [*]Real_t,
               numElem: Index_t, eosvmax: Real_t) void
{
  var idx: Index_t = 0;
  while( idx < numElem ) : ( idx += 1 ) {
    if (vnewc[idx] >= eosvmax) { // impossible due to calling func?
      p_old[idx]        = ZERO;
      compression[idx]  = ZERO;
      compHalfStep[idx] = ZERO;
    }
  }
}

fn EvalEosResetWork(work: [*]Real_t, numElem: Index_t) void
{
  var idx: Index_t = 0;
  while( idx < numElem ) : ( idx += 1 ) {
    work[idx] = ZERO;
  }
}

fn UpdatePE(p: [*]Real_t, p_new: [*]Real_t, e: [*]Real_t, e_new: [*]Real_t,
            q: [*]Real_t, q_new: [*]Real_t, numElem: Index_t) void
{
  var idx: Index_t = 0;
  while( idx < numElem ) : ( idx += 1 ) {
    p[idx] = p_new[idx];
    e[idx] = e_new[idx];
    q[idx] = q_new[idx];
  }
}

fn EvalEOSForElems(domain: *Domain, vnewc: [*]Real_t, numElem: Index_t) void
{
   const  e_cut = domain.e_cut;
   const  p_cut = domain.p_cut;
   const  q_cut = domain.q_cut;

   const eosvmax = domain.eosvmax;
   const eosvmin = domain.eosvmin;
   const pmin    = domain.pmin;
   const emin    = domain.emin;
   const rho0    = domain.refdens;

   const delvc: [*]Real_t = domain.delv;
   const p_old: [*]Real_t = domain.p_old;
   const compression: [*]Real_t = domain.compression;
   const compHalfStep: [*]Real_t = domain.compressionHalfStep;
   const work: [*]Real_t = domain.work;
   const p_new: [*]Real_t = domain.p_new;
   const e_new: [*]Real_t = domain.e_new;
   const q_new: [*]Real_t = domain.q_new;
   const bvc: [*]Real_t = domain.bvc;
   const pbvc: [*]Real_t = domain.pbvc;
   const pHalfStep: [*]Real_t = domain.p_HalfStep;

   EvalCopy(p_old, domain.p, numElem);

   EvalCompression(compression, compHalfStep, numElem, vnewc, delvc);

   // Check for v > eosvmax or v < eosvmin
   if ( eosvmin != ZERO ) {
      EvalEosVmin(vnewc, compHalfStep, compression, numElem, eosvmin);
   }

   if ( eosvmax != ZERO ) {
      EvalEosVmax(vnewc, p_old, compHalfStep, compression, numElem, eosvmax);
   }

   EvalEosResetWork(work, numElem);

   CalcEnergyForElems(p_new, e_new, q_new, bvc, pbvc,
                 p_old, domain.e,  domain.q, compression, compHalfStep,
                 vnewc, work,  delvc, pmin,
                 p_cut, e_cut, q_cut, emin,
                 domain.qq, domain.ql, rho0, eosvmax,
                 pHalfStep, numElem);


   UpdatePE(domain.p, p_new, domain.e, e_new, domain.q, q_new, numElem);

   CalcSoundSpeedForElems(numElem, domain.ss,
             vnewc, rho0, e_new, p_new, pbvc, bvc);
}

fn VolErr3(vnewc: [*]Real_t, vnew: [*]Real_t, v: [*]Real_t, numElem: Index_t,
           eosvmin: Real_t, eosvmax: Real_t) bool
{
  var idx: Index_t = 0;
  while( idx < numElem ) : ( idx += 1 ) {
    vnewc[idx] = vnew[idx];
  }

  if (eosvmin != ZERO) {
    idx = 0;
    while( idx < numElem ) : ( idx += 1 ) {
      if (vnewc[idx] < eosvmin)
        vnewc[idx] = eosvmin;
    }
  }

  if (eosvmax != ZERO) {
    idx = 0;
    while( idx < numElem ) : ( idx += 1 ) {
      if (vnewc[idx] > eosvmax)
        vnewc[idx] = eosvmax;
    }
  }

  idx = 0;
  while( idx < numElem ) : ( idx += 1 ) {
    var vc: Real_t = v[idx];
    if (eosvmin != ZERO) {
      if (vc < eosvmin)
        vc = eosvmin;
    }
    if (eosvmax != ZERO) {
      if (vc > eosvmax)
        vc = eosvmax;
    }
    if (vc <= ZERO) return true;
  }
  return false;
}

fn ApplyMaterialPropertiesForElems(domain: *Domain) void
{
  const numElem = domain.numElem;

  if (numElem != 0) {
    // Expose all of the variables needed for material evaluation

    // Legacy comment, to see this app is simplified from original:
    // Create a domain length (not material length) temporary.
    // we are assuming here that the number of dense ranges is
    // much greater than the number of sigletons.  We are also
    // assuming it is ok to allocate a domain length temporary
    // rather than a material length temporary.

    if (VolErr3(domain.vnewc, domain.vnew, domain.v, numElem,
                domain.eosvmin, domain.eosvmax))
       std.process.exit(@intFromEnum(Err.VolumeError));

    EvalEOSForElems(domain, domain.vnewc, numElem);
  }
}

fn UpdateVolumesForElems(vnew: [*]Real_t, v: [*]Real_t,
                         v_cut: Real_t, length: Index_t) void
{
   if (length != 0) {
      var k: Index_t = 0;
      while ( k < length ) : ( k += 1 ) {
         var tmpV: Real_t = vnew[k];

         if ( @abs(tmpV - ONE) < v_cut )
            tmpV = ONE;

         v[k] = tmpV;
      }
   }

   return;
}

fn LagrangeElements(domain: *Domain, numElem: Index_t) void
{
  CalcLagrangeElements(domain);

  // Calculate Q.  (Monotonic q option requires communication)
  CalcQForElems(domain);

  ApplyMaterialPropertiesForElems(domain);

  UpdateVolumesForElems(domain.vnew, domain.v,
                        domain.v_cut, numElem);
}

fn CalcCourantConstraintForElems(length: Index_t, ss: [*]Real_t,
                                 vdov: [*]Real_t, arealg: [*]Real_t,
                                 qqc: Real_t, dtcourant: *Real_t) void
{
   var dtcourant_tmp: Real_t = 1.0e+20;
   var courant_elem: bool = false;
   // var courant_elem: Index_t = -1;

   const qqc2 = SIXTYFOUR * qqc * qqc;

   var indx: Index_t = 0;
   while ( indx < length) : ( indx += 1 ) {

      var dtf: Real_t = ss[indx] * ss[indx];

      if ( vdov[indx] < ZERO ) {

         dtf = dtf
            + qqc2 * arealg[indx] * arealg[indx] * vdov[indx] * vdov[indx];
      }

      dtf = math.sqrt( dtf );

      dtf = arealg[indx] / dtf;

      // determine minimum timestep with its corresponding elem

      if (vdov[indx] != ZERO) {
         if ( dtf < dtcourant_tmp ) {
            dtcourant_tmp = dtf;
            courant_elem = true;
            // courant_elem = indx;
         }
      }
   }

   // Don't try to register a time constraint if none of the elements
   // were active

   if (courant_elem) { // != -1
      dtcourant.* = dtcourant_tmp;
   }

   return;
}

fn CalcHydroConstraintForElems(length: Index_t, vdov: [*]Real_t,
                               dvovmax: Real_t, dthydro: *Real_t) void
{
   var dthydro_tmp: Real_t = 1.0e+20;
   var hydro_elem: bool = false;
   // var hydro_elem: Index_t = -1;

   var indx: Index_t = 0;
   while ( indx < length) : ( indx += 1 ) {
      if (vdov[indx] != ZERO) {
         const dtdvov = dvovmax / (@abs(vdov[indx])+1.0e-20);
         if ( dthydro_tmp > dtdvov ) {
            dthydro_tmp = dtdvov;
            hydro_elem = true;
            // hydro_elem = indx;
         }
      }
   }

   if (hydro_elem) { // != -1
      dthydro.* = dthydro_tmp;
   }

   return ;
}

fn CalcTimeConstraintsForElems(domain: *Domain) void
{
   CalcCourantConstraintForElems(domain.numElem, domain.ss,
                                 domain.vdov, domain.arealg,
                                 domain.qqc, & domain.dtcourant);

   CalcHydroConstraintForElems(domain.numElem, domain.vdov,
                               domain.dvovmax, & domain.dthydro);
}

fn LagrangeLeapFrog(domain: *Domain) void
{
   // calculate nodal forces, accelerations, velocities, positions, with
   // applied boundary conditions and slide surface considerations
   LagrangeNodal(domain);

   // calculate element quantities (i.e. velocity gradient & q), and update
   // material states
   LagrangeElements(domain, domain.numElem);

   CalcTimeConstraintsForElems(domain);
}

pub fn main() !void
{
   var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
   defer arena.deinit();

   const allocator = arena.allocator();

   const edgeElems : Index_t = 20;
   const edgeNodes : Index_t = edgeElems + 1;
   const domElems  : Index_t = edgeElems*edgeElems*edgeElems;
   const domNodes  : Index_t = edgeNodes*edgeNodes*edgeNodes;

   // ****************************
   // *   Initialize Sedov Mesh  *
   // ****************************

   var domain = Domain {

   // construct a uniform box for this processor

      .sizeX = edgeElems,
      .sizeY = edgeElems,
      .sizeZ = edgeElems,
      .numElem = domElems,
      .numNode = domNodes,

   // *************************
   // * allocate field memory *
   // *************************
   
   // *****************
   // * Elem-centered *
   // *****************

   // elemToNode connectivity
      .nodelist = try allocator.create([domElems][8]Index_t),

   // elem connectivity through face
      .lxim = try allocator.create([domElems]Index_t),
      .lxip = try allocator.create([domElems]Index_t),
      .letam = try allocator.create([domElems]Index_t),
      .letap = try allocator.create([domElems]Index_t),
      .lzetam = try allocator.create([domElems]Index_t),
      .lzetap = try allocator.create([domElems]Index_t),

   // elem face symm/free-surface flag
      .elemBC = try allocator.create([domElems]u32),

      .e = try allocator.create([domElems]Real_t), // energy
      .p = try allocator.create([domElems]Real_t), // pressure

      .q = try allocator.create([domElems]Real_t),  // artificial viscosity
      .ql = try allocator.create([domElems]Real_t), //   linear term
      .qq = try allocator.create([domElems]Real_t), //   quadratic term

      .v = try allocator.create([domElems]Real_t),    // relative volume
      .volo = try allocator.create([domElems]Real_t), // reference volume
      .delv = try allocator.create([domElems]Real_t), // vnew - v
      .vdov = try allocator.create([domElems]Real_t), // volume deriv over volume

      .arealg = try allocator.create([domElems]Real_t), // characteristic length

      .ss = try allocator.create([domElems]Real_t),     // "sound speed"

      .elemMass = try allocator.create([domElems]Real_t), // mass

   // *****************
   // * Node-centered *
   // *****************

      .x = try allocator.create([domNodes]Real_t), // coordinates
      .y = try allocator.create([domNodes]Real_t),
      .z = try allocator.create([domNodes]Real_t),

      .xd = try allocator.create([domNodes]Real_t), // velocities
      .yd = try allocator.create([domNodes]Real_t),
      .zd = try allocator.create([domNodes]Real_t),

      .xdd = try allocator.create([domNodes]Real_t), // accelerations
      .ydd = try allocator.create([domNodes]Real_t),
      .zdd = try allocator.create([domNodes]Real_t),

      .fx = try allocator.create([domNodes]Real_t), // forces
      .fy = try allocator.create([domNodes]Real_t),
      .fz = try allocator.create([domNodes]Real_t),

      .nodalMass = try allocator.create([domNodes]Real_t), // mass

   // Boundary nodesets

      .symmX = try allocator.create([edgeNodes*edgeNodes]Index_t),
      .symmY = try allocator.create([edgeNodes*edgeNodes]Index_t),
      .symmZ = try allocator.create([edgeNodes*edgeNodes]Index_t),

      .dxx = try allocator.create([domElems]Real_t), // principal strains
      .dyy = try allocator.create([domElems]Real_t),
      .dzz = try allocator.create([domElems]Real_t),

      .delv_xi = try allocator.create([domElems]Real_t), // velocity gradient
      .delv_eta = try allocator.create([domElems]Real_t),
      .delv_zeta = try allocator.create([domElems]Real_t),

      .delx_xi = try allocator.create([domElems]Real_t), // position gradient
      .delx_eta = try allocator.create([domElems]Real_t),
      .delx_zeta = try allocator.create([domElems]Real_t),

      .sigxx = try allocator.create([domElems]Real_t),
      .sigyy = try allocator.create([domElems]Real_t),
      .sigzz = try allocator.create([domElems]Real_t),

      .dvdx = try allocator.create([domElems][8]Real_t),
      .dvdy = try allocator.create([domElems][8]Real_t),
      .dvdz = try allocator.create([domElems][8]Real_t),

      .x8n = try allocator.create([domElems][8]Real_t),
      .y8n = try allocator.create([domElems][8]Real_t),
      .z8n = try allocator.create([domElems][8]Real_t),

      .determ = try allocator.create([domElems]Real_t),
      .vnew = try allocator.create([domElems]Real_t),
      .vnewc = try allocator.create([domElems]Real_t),

      .p_old = try allocator.create([domElems]Real_t),
      .p_new = try allocator.create([domElems]Real_t),
      .p_HalfStep = try allocator.create([domElems]Real_t),
      .q_new = try allocator.create([domElems]Real_t),
      .e_new = try allocator.create([domElems]Real_t),
      .bvc = try allocator.create([domElems]Real_t),
      .pbvc = try allocator.create([domElems]Real_t),
      .work = try allocator.create([domElems]Real_t),

      .compression = try allocator.create([domElems]Real_t),
      .compressionHalfStep = try allocator.create([domElems]Real_t),

   // initialize material parameters

      .dtfixed   = -1.0e-7,
      .deltatime = 1.0e-7,
      .deltatimemultlb = 1.1,
      .deltatimemultub = 1.2,
      .stoptime  = 1.0e-2,
      .dtcourant = 1.0e+20,
      .dthydro   = 1.0e+20,
      .dtmax     = 1.0e-2,
      .time      = ZERO,
      .cycle     = 0,

      .e_cut     = 1.0e-7,
      .p_cut     = 1.0e-7,
      .q_cut     = 1.0e-7,
      .u_cut     = 1.0e-7,
      .v_cut     = 1.0e-10,

      .hgcoef    = 3.0,

      .qstop              = 1.0e+12,
      .monoq_max_slope    = ONE,
      .monoq_limiter_mult = 2.0,
      .qlc_monoq          = HALF,
      .qqc_monoq          = 2.0/3.0,
      .qqc                = 2.0,

      .pmin      =  ZERO,
      .emin      = -1.0e+15,

      .dvovmax   =  0.1,

      .eosvmax   =  1.0e+9,
      .eosvmin   =  1.0e-9,

      .refdens   =  ONE
   } ;

   // Basic Field Initialization

   var k: Index_t = 0;
   while ( k < domElems ) : ( k += 1 ) {
      domain.e[k] = ZERO;
      domain.p[k] = ZERO;
      domain.q[k] = ZERO;
      domain.v[k] = ONE;
   }

   k = 0;
   while ( k < domNodes ) : ( k += 1 ) {
      domain.xd[k] = ZERO;
      domain.yd[k] = ZERO;
      domain.zd[k] = ZERO;
   }

   k = 0;
   while ( k < domNodes ) : ( k += 1 ) {
      domain.xdd[k] = ZERO;
      domain.ydd[k] = ZERO;
      domain.zdd[k] = ZERO;
   }

   // initialize nodal coordinates

   var nidx: Index_t = 0;
   var tz: Real_t = ZERO;
   var plane: Index_t = 0;
   while ( plane < edgeNodes ) : ( plane += 1 ) {
      var ty: Real_t = ZERO;
      var row: Index_t = 0;
      while ( row < edgeNodes ) : ( row += 1 ) {
         var tx: Real_t = ZERO;
         var col: Index_t = 0;
         while ( col < edgeNodes ) : ( col += 1 ) {
            domain.x[nidx] = tx;
            domain.y[nidx] = ty;
            domain.z[nidx] = tz;
            nidx += 1;
            // tx += ds; /* may accumulate roundoff... */
            tx = 1.125*IndexToReal(col+1)/IndexToReal(edgeElems);
         }
         // ty += ds;  /* may accumulate roundoff... */
         ty = 1.125*IndexToReal(row+1)/IndexToReal(edgeElems);
      }
      // tz += ds;  /* may accumulate roundoff... */
      tz = 1.125*IndexToReal(plane+1)/IndexToReal(edgeElems);
   }


   // embed hexehedral elements in nodal point lattice

   nidx = 0;
   var zidx: Index_t = 0;
   plane = 0;
   while ( plane < edgeElems ) : ( plane += 1 ) {
      var row: Index_t = 0;
      while ( row < edgeElems ) : ( row += 1 ) {
         var col: Index_t = 0;
         while ( col < edgeElems ) : ( col += 1 ) {
            // var localNode = domain.nodelist[zidx];
            // localNode[0] = nidx                                       ;
            // localNode[1] = nidx                                   + 1 ;
            // localNode[2] = nidx                       + edgeNodes + 1 ;
            // localNode[3] = nidx                       + edgeNodes     ;
            // localNode[4] = nidx + edgeNodes*edgeNodes                 ;
            // localNode[5] = nidx + edgeNodes*edgeNodes             + 1 ;
            // localNode[6] = nidx + edgeNodes*edgeNodes + edgeNodes + 1 ;
            // localNode[7] = nidx + edgeNodes*edgeNodes + edgeNodes     ;
            domain.nodelist[zidx][0] = nidx                                       ;
            domain.nodelist[zidx][1] = nidx                                   + 1 ;
            domain.nodelist[zidx][2] = nidx                       + edgeNodes + 1 ;
            domain.nodelist[zidx][3] = nidx                       + edgeNodes     ;
            domain.nodelist[zidx][4] = nidx + edgeNodes*edgeNodes                 ;
            domain.nodelist[zidx][5] = nidx + edgeNodes*edgeNodes             + 1 ;
            domain.nodelist[zidx][6] = nidx + edgeNodes*edgeNodes + edgeNodes + 1 ;
            domain.nodelist[zidx][7] = nidx + edgeNodes*edgeNodes + edgeNodes     ;
            zidx += 1;
            nidx += 1;
         }
         nidx += 1;
      }
      nidx += edgeNodes;
   }

   // initialize field data
   k = 0;
   while ( k < domNodes ) : ( k += 1 ) {
      domain.nodalMass[k] = ZERO;
   }

   var x_local: [8]Real_t = undefined;
   var y_local: [8]Real_t = undefined;
   var z_local: [8]Real_t = undefined;

   k = 0;
   while ( k < domElems ) : ( k += 1 ) {
      const elemToNode = domain.nodelist[k];
      var lnode: Index_t = 0;
      while ( lnode < 8 ) : ( lnode += 1 ) {
        const gnode = elemToNode[lnode];
        x_local[lnode] = domain.x[gnode];
        y_local[lnode] = domain.y[gnode];
        z_local[lnode] = domain.z[gnode];
      }

      // volume calculations
      const volume = CalcElemVolume(x_local, y_local, z_local);
      domain.volo[k] = volume;
      domain.elemMass[k] = volume;
      var j: Index_t = 0;
      while ( j < 8 ) : ( j += 1 ) {
         const idx = elemToNode[j];
         domain.nodalMass[idx] += volume / 8.0;
      }
   }

   // deposit energy
   domain.e[0] = 3.948746e+7;

   // set up symmetry nodesets
   nidx = 0;
   k = 0;
   while ( k < edgeNodes ) : ( k += 1 ) {
      const planeInc = k*edgeNodes*edgeNodes;
      const rowInc   = k*edgeNodes;
      var j: Index_t = 0;
      while ( j < edgeNodes ) : ( j += 1 ) {
         domain.symmX[nidx] = planeInc + j*edgeNodes;
         domain.symmY[nidx] = planeInc + j;
         domain.symmZ[nidx] = rowInc   + j;
         nidx += 1;
      }
   }

   // set up elemement connectivity information
   domain.lxim[0] = 0;
   k = 1;
   while ( k < domElems ) : ( k += 1 ) {
      domain.lxim[k]   = k-1;
      domain.lxip[k-1] = k;
   }
   domain.lxip[domElems-1] = domElems-1;

   k = 0;
   while ( k < edgeElems ) : ( k += 1 ) {
      domain.letam[k] = k; 
      domain.letap[domElems-edgeElems+k] = domElems-edgeElems+k;
   }
   k = edgeElems;
   while ( k < domElems ) : ( k += 1 ) {
      domain.letam[k] = k-edgeElems;
      domain.letap[k-edgeElems] = k;
   }

   k = 0;
   while ( k < edgeElems*edgeElems ) : ( k += 1 ) {
      domain.lzetam[k] = k;
      domain.lzetap[domElems-edgeElems*edgeElems+k] =
         domElems-edgeElems*edgeElems+k;
   }
   k = edgeElems*edgeElems;
   while ( k < domElems ) : ( k += 1 ) {
      domain.lzetam[k] = k - edgeElems*edgeElems;
      domain.lzetap[k-edgeElems*edgeElems] = k;
   }

   // set up boundary condition information
   k = 0;
   while ( k < domElems ) : ( k += 1 ) {
      domain.elemBC[k] = 0;  // clear BCs by default
   }

   // faces on "external" boundaries will be
   // symmetry plane or free surface BCs
   k = 0;
   while ( k < edgeElems ) : ( k += 1 ) {
      const planeInc2 = k*edgeElems*edgeElems;
      const rowInc2   = k*edgeElems;
      var j: Index_t = 0;
      while ( j < edgeElems ) : ( j += 1 ) {
         domain.elemBC[planeInc2+j*edgeElems] |=
            BCbits(BC.XI_M_SYMM);
         domain.elemBC[planeInc2+j*edgeElems+edgeElems-1] |=
            BCbits(BC.XI_P_FREE);
         domain.elemBC[planeInc2+j] |=
            BCbits(BC.ETA_M_SYMM);
         domain.elemBC[planeInc2+j+edgeElems*edgeElems-edgeElems] |=
            BCbits(BC.ETA_P_FREE);
         domain.elemBC[rowInc2+j] |=
            BCbits(BC.ZETA_M_SYMM);
         domain.elemBC[rowInc2+j+domElems-edgeElems*edgeElems] |=
            BCbits(BC.ZETA_P_FREE);
      }
   }

   // timestep to solution
   while(domain.time < domain.stoptime) {
      TimeIncrement(&domain);
      LagrangeLeapFrog(&domain);
      try stdout.print("{e:9.7} {e:9.7}\n",
                        .{ domain.time, domain.deltatime });
   }

   return;
}
