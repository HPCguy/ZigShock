//*************************************************************************
// Program:  ShockTube.C
// Purpose:  1D shock tube, split flux Euler equations
//
//         | m  |            |    mv    |
//     Q = | mv |        F = | mv^2 + P |
//         | E  |            |  v(E+P)  |
//
//     P = (gamma - 1.0)[E - 0.5 mv^2 ]
//
//             Cp
//     gamma = --     m = mass/volume   v = velocity
//             Cv
//
//     All quantities are non-dimensionalized.
//
//     @Q   @F    @Q   @F @Q
//     -- + -- =  -- + -- -- = 0
//     @t   @x    @t   @Q @x
//
//************************************************************************/

const std = @import("std");
const stdout = std.io.getStdOut().writer();
const math = std.math;

const Real_t = f32;
const Index_t = usize;

fn IndexToReal(idx: Index_t) Real_t
{
   return @floatFromInt(idx);
}

const gammaa:       Real_t = 1.41421356237;
const gammaInverse: Real_t = 0.707106781187;

const ZERO:         Real_t = 0.0;
const HALF:         Real_t = 0.5;
const ONE:          Real_t = 1.0;

//*************************************************************************
// Subroutine:  CreateShockTubeMesh
// Purpose   :  Build an empty mesh for the shock tube
//
//
//    Gaps between elements are faces
//                   |
//    -------------------------------
//    |   |   |             |   |   |
//
// ### ### ### ###       ### ### ### ###
// ### ### ### ###  ...  ### ### ### ###  <--- 1D Shock tube model
// ### ### ### ###       ### ### ### ###
//
//  |  |                           |  |
//  |  -----------------------------  |
// Inflow           |               Outflow
// Element      Tube Elements       Element


//************************************************************************/

fn InitializeShockTubeMesh(numElem: Index_t, mass: [*]Real_t,
                           momentum:[*]Real_t, energy:[*]Real_t,
                           pressure:[*]Real_t) void
{
   const midTube = numElem / 2;

   var massInitial: Real_t     = ONE;
   var pressureInitial: Real_t = gammaInverse;
   var energyInitial: Real_t   = pressureInitial/(gammaa-ONE);

   var idx: Index_t = 0;
   while ( idx < midTube ) : ( idx += 1 ) {
      mass[idx]     = massInitial;
      pressure[idx] = pressureInitial;
      energy[idx]   = energyInitial;
   }

   const pressureRatio: Real_t = 0.4;
   const densityRatio:  Real_t = 0.7;

   massInitial     = massInitial * densityRatio;
   pressureInitial = pressureInitial * pressureRatio;
   energyInitial   = pressureInitial/(gammaa - ONE);

   idx = midTube;
   while ( idx < numElem ) : ( idx += 1 ) {
      mass[idx]     = massInitial;
      pressure[idx] = pressureInitial;
      energy[idx]   = energyInitial;
   }

   idx = 0;
   while ( idx < numElem ) : ( idx += 1 ) {
      momentum[idx] = ZERO;
   }
}


//*************************************************************************
// Subroutine:  ComputeFaceInfo
// Purpose   :  Compute F quantities at faces.
//
//  @F   @F0   @F1   @F2
//  -- = --- + --- + ---
//  @x   @x    @x    @x
//
//  Calculate F0, F1 and F2 at the face centers.
//
//************************************************************************/

fn ComputeFaceInfo(numFace: Index_t,
                   mass: [*]const Real_t, momentum: [*]const Real_t,
                   energy: [*]const Real_t,
                   f0: [*]Real_t, f1: [*]Real_t, f2: [*]Real_t) void
{
   var idx: Index_t = 0;
   while ( idx < numFace ) : ( idx += 1 ) {
      // each face has an upwind and downwind element
      const upWind   = idx;     // upwind element
      const downWind = idx + 1; // downwind element

      // calculate face centered quantities
      var massf: Real_t =     HALF * (mass[upWind]     + mass[downWind]);
      var momentumf: Real_t = HALF * (momentum[upWind] + momentum[downWind]);
      var energyf: Real_t =   HALF * (energy[upWind]   + energy[downWind]);
      var pressuref: Real_t = (gammaa - ONE) *
                      (energyf - HALF*momentumf*momentumf/massf);
      const c: Real_t = math.sqrt(gammaa*pressuref/massf);
      const v = momentumf/massf;

      // Now that we have the wave speeds, we might want to
      // look for the max wave speed here, and update dt
      // appropriately right before leaving this function.
      // ...

      // OK, calculate face quantities

      const contributor = if (v >= ZERO) upWind else downWind;
      massf = mass[contributor];
      momentumf = momentum[contributor];
      energyf = energy[contributor];
      pressuref = energyf - HALF*momentumf*momentumf/massf;
      const ev = v*(gammaa - ONE);

      f0[idx] = ev*massf;
      f1[idx] = ev*momentumf;
      f2[idx] = ev*(energyf - pressuref);

      const contributorp = if (v + c >= ZERO) upWind else downWind;
      massf = mass[contributorp];
      momentumf = momentum[contributorp];
      energyf = energy[contributorp];
      pressuref = (gammaa - ONE)*(energyf - HALF*momentumf*momentumf/massf);
      const evp = HALF*(v + c);
      const cLocalp: Real_t = math.sqrt(gammaa*pressuref/massf);

      f0[idx] += evp*massf;
      f1[idx] += evp*(momentumf + massf*cLocalp);
      f2[idx] += evp*(energyf + pressuref + momentumf*cLocalp);

      const contributorm = if (v - c >= ZERO) upWind else downWind;
      massf = mass[contributorm];
      momentumf = momentum[contributorm];
      energyf = energy[contributorm];
      pressuref = (gammaa - ONE)*(energyf - HALF*momentumf*momentumf/massf);
      const evm = HALF*(v - c);
      const cLocalm: Real_t = math.sqrt(gammaa*pressuref/massf);

      f0[idx] += evm*massf;
      f1[idx] += evm*(momentumf - massf*cLocalm);
      f2[idx] += evm*(energyf + pressuref - momentumf*cLocalm);
   }
}


//*************************************************************************
// Subroutine:  UpdateElemInfo
// Purpose   :  Q(elem) = Q(elem) + deltaQ(elem)
//
//  deltaQ(elem) = - (F(downWindFace) - F(upWindFace)) * dt / dx ;
//
//************************************************************************/

fn UpdateElemInfo(numElem: Index_t,
                  mass: [*]Real_t, momentum: [*]Real_t, energy: [*]Real_t,
                  pressure: [*]Real_t, f0: [*]const Real_t,
                  f1: [*]const Real_t, f2: [*]const Real_t, dtdx: Real_t) void
{
   var idx: Index_t = 1;
   while ( idx < numElem ) : ( idx += 1 ) {
      // each element inside the tube has an upwind and downwind face
      const upWind = idx-1;     // upwind face
      const downWind = idx;   // downwind face

      mass[idx]     -= gammaInverse*(f0[downWind] - f0[upWind])*dtdx;
      momentum[idx] -= gammaInverse*(f1[downWind] - f1[upWind])*dtdx;
      energy[idx]   -= gammaInverse*(f2[downWind] - f2[upWind])*dtdx;
      pressure[idx]  = (gammaa - ONE) *
                       ( energy[idx]
                       - HALF*momentum[idx]*momentum[idx]/mass[idx]);
   }
}


//*************************************************************************
// Subroutine:  DumpField
// Purpose   :  Create a plot for a single field
//************************************************************************/

fn DumpField(tag: []const u8, numElem: Index_t, field: [*]const Real_t) !void
{
   try stdout.print("{s}\n", .{ tag });
   var idx: Index_t = 0;
   while( idx < numElem ) : ( idx += 1 ) {
      try stdout.print("{d}.0 {e:9.6}\n", .{ idx, field[idx] });
   }
   try stdout.print("\n\n", .{});
}


//*************************************************************************
// Subroutine:  DumpPlot
// Purpose   :  create output that can be viewed with gnuplot: plot "file"
//************************************************************************/

fn DumpPlot(numElem: Index_t, mass: [*]const Real_t, momentum: [*]const Real_t,
                              energy: [*]const Real_t,
                              pressure: [*]const Real_t) !void
{
   const m = "# mass";
   try DumpField(m, numElem, mass);
   const mom = "# momentum";
   try DumpField(mom, numElem, momentum);
   const e = "# energy";
   try DumpField(e, numElem, energy);
   const p = "# pressure";
   try DumpField(p, numElem, pressure);
}


//*************************************************************************
// Subroutine:  main
// Purpose   :  Simulate a 1D Shock Tube using split flux Euler formulation
//************************************************************************/

pub fn main() !void
{
   var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
   defer arena.deinit();
      
   const allocator = arena.allocator();

   const numElems = 8192;          // 2048
   const numFaces = numElems - 1;
   const numTotalCycles = 6144;    // 1024
   // const dumpInterval = 500;

   const mass: [*]Real_t     = try allocator.create([numElems+1]Real_t);
   const momentum: [*]Real_t = try allocator.create([numElems+1]Real_t);
   const energy: [*]Real_t   = try allocator.create([numElems+1]Real_t);
   const pressure: [*]Real_t = try allocator.create([numElems+1]Real_t);

   const f0: [*]Real_t = try allocator.create([numElems-1]Real_t);
   const f1: [*]Real_t = try allocator.create([numElems-1]Real_t);
   const f2: [*]Real_t = try allocator.create([numElems-1]Real_t);

   InitializeShockTubeMesh(numElems+1, mass, momentum, energy, pressure);

   var time: Real_t = ZERO;
   const dx: Real_t = ONE / IndexToReal( numElems );
   const safetyFactor: Real_t = 0.4;
   const dt: Real_t = safetyFactor * dx;

   var currCycle: u32 = 0;
   while ( currCycle < numTotalCycles ) : ( currCycle += 1 ) {
      // if (currCycle % dumpInterval == 0)
      //    try DumpPlot(numElems, mass, momentum, energy, pressure);

      ComputeFaceInfo(numFaces, mass, momentum, energy, f0, f1, f2);
      UpdateElemInfo (numElems-1, mass, momentum, energy, pressure,
                      f0, f1, f2, dt/dx);
      time = time + dt;
   }

   try DumpPlot(numElems, mass, momentum, energy, pressure);

   return ;
}

