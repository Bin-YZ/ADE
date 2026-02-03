# support math and numpy
import sys # used for exiting in case of critical errors
import logging # use logging for output, debugging etc..
"""
We use python "logging" for control of output! 
DEBUG 	Detailed information, typically of interest only when diagnosing problems.
INFO 	Confirmation that things are working as expected.
WARNING 	An indication that something unexpected happened, or indicative of some problem in the near future (e.g. ‘disk space low’). The software is still working as expected.
ERROR 	Due to a more serious problem, the software has not been able to perform some function.
CRITICAL 	A serious error, indicating that the program itself may be unable to continue running.
"""
import math
import numpy as np
import pandas as pd

# load dolfinx stuff
from dolfinx import mesh,fem,io,plot,cpp,log,geometry
# petsc solver related
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc
#from basix.ufl import FiniteElement
import basix.ufl
# this is for ufl..i.e. weak form definition
from ufl import (as_ufl, CellDiameter, CellVolume, FacetArea,FacetNormal, Measure, SpatialCoordinate, TestFunction, 
                TrialFunction, avg, min_value, div, dot, jump, dx, dS, grad, inner, lhs, rhs,transpose)

from dolfinx.plot import vtk_mesh
from dolfinx.io import VTXWriter # for lagrange and discontinuous lagrange/Galerkin elements the VTXWriter is supported
from dolfinx.io import XDMFFile  # supports only lagrange elements of various orders
# support of parallelisation 
from mpi4py import MPI
from petsc4py import PETSc


################################## extra class for boundary conditions used in ADE_DG #######################3
class BoundaryCondition(): 
    """
    this class is used to process a list with boundary conditions and insert boundary conditions in the weak form 
    It follows the FEniCSx tutorial on setting multiple boundary conditions.

    Note that locate_dofs_topological does not work with discontinuous Galerkin elements 
    """
    def __init__(self, problem,type, marker, values):
        self._type = type
        
        if type == "Dirichlet": # fill into bcs[]
            u_D = fem.Constant(problem.mymesh, values)    # for constant value at boundary only...no function supported here
#            u_D = fem.Function(self.V)
#            u_D.interpolate(values)
            #facets = self.facet_tag.find(marker)
#           Unfortunately locate_dofs_topological seems not to work with DG as dofs are not part of the facet (but part of the cell!) -> https://fenicsproject.discourse.group/t/dirichlet-boundary-of-dg-rt-n1curl-element/12584
#           it does not find the dofs (nodes/gauss points)            
            #dofs = fem.locate_dofs_topological(V=self.V, entity_dim=self.fdim, entities=facets,remote=True)
# instead of giving marker id, we give the function call for left or right
            dofs = fem.locate_dofs_geometrical(problem.V, marker)
            self._bc = fem.dirichletbc(u_D, dofs,problem.V)            
            logging.debug("Dirichlet boundary: dofs:  "+str(dofs))
        elif type == "Neumann":  # add to weak form
            self._bc = - dt * inner(values, v) * ds(marker)
        elif type == "Robin": # add to wak form
            self._bc = values[0] * inner(u-values[1], v)* ds(marker)
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(type))
    @property
    def bc(self):
        return self._bc

    @property
    def type(self):
        return self._type

##################################### class for ADE solver ################################33

class ADE_DG():
#    __docformat__ = "numpy" 
    """
    ADE_DG is FEniCS based Adevecton-Diffusion Equation (ADE) Solver class specifically tailored for evaluation of core scale tracer diffusion experiments. 
    This solver supports only single component tracer transport.
    
    The ADE is solved based on discontiuous Lagrange (discontinous Galerkin) Finite Elements and using symmetric interior weighted penalty (DG-SWIP) method as explained in Ern et al. (2008).
    One should note, that convergence and accuracy of the solution is highly sensitive to the choice of penalty parameter. We use the weighting and penalty parameters provided in Kempf (2019) and Etangsale et al. (2021).
    
    Ern, A., Stephansen, A. F., & Zunino, P. (2008). A discontinuous Galerkin method with weighted averages for advection-diffusion equations with locally small and anisotropic diffusivity. IMA  Journal of Numerical Analysis, 29(2), 235–256.https://doi.org/10.1093/imanum/drm050
    
    Etangsale, G., Fontaine, V., & Rajaonison, N. (2021). Performances of Hybridized-, Embedded-, and Weighted-Interior Penalty Discontinuous Galerkin Methods for Heterogeneous and Anisotropic Diffusion Problems. Frontiers in Water, 3. https://doi.org/10.3389/frwa.2021.716459
    
    Kempf, D. (2019). Code Generation for High Performance PDE Solvers on Modern Architectures. Heidelberg University Library. https://doi.org/10.11588/heidok.00027360
    """
    
    def __init__(self, mesh_fname, deg=1):
        """
        initialisation of FE spaces and functions for material parameters

        Parameters
        ----------
        
        mymesh_fname: 
            A string that contains the name of a gmsh mesh file. 
            
            The mesh file should contain (physical) material markers for influx and outflux boundaries, and for filter and sample materials. Only 2D meshes were tested so far. 

        deg:
            Integer degree for DG elements (defaults to 1) 
            0: finite volume scheme.
            1: standard value
            2: higher accuracy buy doubling calculation points
        """

        ############## set logging level code ###################
        logging.basicConfig(level=logging.INFO,
                    force = True)

        ##################  all kind of variables #####################
        
        logging.info("INFO: init with: "+str(mesh_fname)+" deg "+str(deg)) 
        
        self.__deg=deg # indicate it is private -> changing this after initialisation is not useful!
        
         # read gmsh file
        # for import of 2D meshes from gmsh the gdim argument is required, as gmsh always creates meshes in 3D space...2D mesh in 3D space will cause problems with weak form (ufl): as fenics will provide 2D vectors/tensors and ufl defines 3D vector/tensors 
        self.mymesh, self.cell_tag, self.facet_tag =io.gmshio.read_from_msh(mesh_fname,MPI.COMM_WORLD,gdim=2)
        
        self.cell = str(self.mymesh.ufl_cell())
        
        self.tdim=self.mymesh.topology.dim 
        self.fdim=self.mymesh.topology.dim -1 # faces for boundary conditon have 1 dimension less than mesh
        self.mymesh.topology.create_connectivity(self.fdim, self.tdim)

        logging.info("INFO: mesh read! mesh dimension: "+str(self.mymesh.topology.dim)+" boundary dimension "+str(self.fdim)+" geometry dimension: "+str(self.mymesh.geometry.dim))
        # define FE element spaces for calculation, output and cellwise constant materials
        self.v_dg =basix.ufl.element("DG", self.cell, self.__deg) # DG is discontinous galerkin 
        self.v_cg = basix.ufl.element("CG", self.cell, 1)  # CG used for interpolated output
        # the next lines are hardcoded for triangular meshes
        self.v_dg0t = basix.ufl.element("DG", self.cell, 0,shape=(self.mymesh.geometry.dim,self.mymesh.geometry.dim),symmetry=True, dtype=PETSc.RealType) # DG with Deg 0 is discontinous galerkin for cell based tensors
        self.v_dg0v = basix.ufl.element("DG", self.cell, 0,shape=(self.mymesh.geometry.dim,),dtype=PETSc.RealType) # DG  with Deg 0 is discontinous galerkin for cell based vectors
        self.v_dg0 = basix.ufl.element("DG", self.cell, 0) # DG  with Deg 0 is discontinous galerkin for cell based scalars
        
        
        # Define function spaces
        # V is function space for calculations
        self.V = fem.functionspace(self.mymesh, self.v_dg ) # V is function space for calculations
        
        # Define function space for cellwise constant parameters
        self.V0 = fem.functionspace(self.mymesh, self.v_dg0 )
        
        # cellwise constant function space for diffusion tensor ...newest way to do in fenicsx is via shape parameter
        # https://fenicsproject.discourse.group/t/different-material-properties-in-subdomains-tensor-instead-of-scalar/9887/7
        self.V0T = fem.functionspace(self.mymesh, self.v_dg0t )
        
        # V1 is for interpolation to lagrange elements used for simple visualization
        self.V1 = fem.functionspace(self.mymesh, self.v_cg)
        
        # V2 is cellwise constant Vector space 
        self.V2 = fem.functionspace(self.mymesh,self.v_dg0v) # vector for e.g. water flux

        # coordinates from ufl package 
        self.x = SpatialCoordinate(self.mymesh)
        
        # inital solution (u_old)
        self.u0 = fem.Function(self.V)
        self.u0.name = "u0"

        #  new_solution
        self.uh = fem.Function(self.V)
        self.uh.name = "uh"
        
        # interpolate solution to lagrange elements for visualization in Paraview -> oscillations during interpolation possible!
        self.__uout = fem.Function(self.V1)
        self.__uout.name = "concentration"

        ########################## material properties #########################
        # material properties are defined for all cells....we make same private, to indicate that messing with them should be done by specific function in a controlled way
        #################################################
        self.__De=fem.Function(self.V0) 
        self.__De.name="De"
        self.__DeT=fem.Function(self.V0T) 
        self.__DeT.name="DeT"
        self.set_DeT() # hidden here, as anisotropy tensor for 1D problems is not needed
        self.__porosity=fem.Function(self.V0)
        self.__porosity.name="porosity"

        # for velocity/advection
        self.__darcy_flux = fem.Function(self.V2)
        self.__darcy_flux.name = "darcy_flux"

        # for Temperature (Added for Arrhenius support)
        self.__Temp = fem.Function(self.V1)  
        self.__Temp.name = "Temperature"
        self.__Temp0 = fem.Function(self.V0)
        self.__Temp0.name = "Temperature_cell"
        self.__Temp_ref = PETSc.RealType(293.15)
        self.__Temp_last = None

        ############# definitions for diffusion cell ##################
        # cross-section
        self.__cross_section_area=PETSc.RealType(1.0)
        # reservoir concentrations are update according to in- or out-fluxes
        self.__flag_reservoirs = True
        # reset up- or down-stream reservoir concentrations 
        self.__flag_exchange_upstream_reservoir=False  # set to True for out-diffusion
        self.__flag_exchange_downstream_reservoir=True # works for in and out-diffusion        
        # for upstream reservoir
        self.__upstream_volume = PETSc.RealType(1.0)
        self.__upstream_concentration=PETSc.RealType(1.0)
        self.__upstream_boundary_id=0
        self.__upstream_boundary_xlocation=0.0
        self.__upstream_exchange_times=np.array([0])  # will be overwritten if times are set
        self.__upstream_exchange_times_done=np.full(len([0]), True) # will be overwritten if times are set
        # for downstream reservoir
        self.__downstream_volume = PETSc.RealType(1.0)
        self.__downstream_concentration=PETSc.RealType(1.0)
        self.__downstream_boundary_id=0
        self.__downstream_boundary_xlocation=1.0
        self.__downstream_exchange_times=np.array([0])  # will be overwritten if times are set
        self.__downstream_exchange_times_done=np.full(len([0]), True) # will be overwritten if times are set

        ############### parameters related to control simulation
        self.__T_end = PETSc.RealType(3600.0*24.0*7.0)   # (s) default 1 week
        self.__dt=PETSc.RealType(1.0)  # (s) 
        self.__dt_max=PETSc.RealType(1.0e6*3600.0*24.0*365.25) # 1 million years.....
        self.__T_start = PETSc.RealType(0.0)
        self.__tol_adaptive=PETSc.RealType(1.0e-3) # tolerance for adaptive time stepping
        self.__theta=PETSc.RealType(0.5)   # theta for time stepping scheme

        ############### output files #####################
        self.__vtx_filename = './output/vtkx_results.pb'
        self.__vtx_intervall = int(0) # zero or less is no ouput!

######################## set logging level #########################
    def set_logging_level(self,level):
        """ sets logging level
        Parameters
        ----------
        level (integer or string) default is 20 or INFO if ADE_DG object is initialized

        Information from logging library:
        ---------------------------------
        
                The numeric values of logging levels are given in the following table. These are primarily of interest if you want to define your own levels, and need them to have specific values relative to the predefined levels. If you define a level with the same numeric value, it overwrites the predefined value; the predefined name is lost.
        
        Level   Numeric value   What it means / When to use it
        
        NOTSET 0        When set on a logger, indicates that ancestor loggers are to be consulted to determine the effective level. If that still resolves to NOTSET, then all events are logged. When set on a handler, all events are handled.
        
        DEBUG 10        Detailed information, typically only of interest to a developer trying to diagnose a problem.
        
        INFO 20         Confirmation that things are working as expected.
        
        WARNING 30      An indication that something unexpected happened, or that a problem might occur in the near future (e.g. ‘disk space low’). The software is still working as expected.
        
        ERROR 40        Due to a more serious problem, the software has not been able to perform some function.
        
        CRITICAL 50     A serious error, indicating that the program itself may be unable to continue running.        
        """
        logger = logging.getLogger()
        logger.setLevel(level)
        print("set logging to level: ",level) # print always!
    def set_logging_disabled(self):
        """ disables all the anoying logging output """
        logger = logging.getLogger()
        logger.disabled = True
        print("Logging is disabled!") # print always!
    def set_logging_enabled(self):
        """ enables the helpful information flood again """
        logger = logging.getLogger()
        logger.disabled = False
############## access to internal data ####################3333
    def set_tol_adaptive(self,value=1.0e-3):
        """
        Sets the error tolerance for fully automated time stepping scheme

        Parameters
        ----------

        tol:
            value (float) 

        """
        self.__tol_adaptive=PETSc.RealType(value)
        
    def set_theta(self,theta=0.5):
        """
        Sets the theta value for time discretization

        Parameters
        ----------
        
        theta:
            value (float) 
            0.5 Implicit (Crank-Nicolson)
            1.0 Full implicit
            0.0 Explicit
        """
        self.__theta=PETSc.RealType(theta)
        
    def set_dt(self,value):
        """
        Sets the time for ending the simulation

        Parameters
        ----------
        
        tend:
            time step size (seconds) 
        """
        self.__dt=PETSc.RealType(value)   

    def set_dtmax(self,value):
        """
        Sets the maximum time step size for adaptive time stepping loop

        Parameters
        ----------
        
        tend:
            time step size (seconds) 
        """
        self.__dt_max=PETSc.RealType(value)   

    
    def set_T_end(self,tend):
        """
        Sets the time for ending the simulation

        Parameters
        ----------
        
        tend:
            time (seconds) 
        """
        self.__T_end=PETSc.RealType(tend)

    def set_update_reservoir_boundaries(self,bflag=True):
        """Sets a flag that enforces usage and update of up- and downstream reservoirs concentrations during simulation.
        The reservoir concentrations are changed according to up- and downstream fluxes.
        The reservoir properties need to be defined!

        Parameters
        ----------
        
        bflag:
            True (default) or False
        """
        self.__flag_reservoirs = bflag
        
    def set_upstream_boundary(self,id,location=0.0):
        """
        Sets the boundary (material) id and the (x-axis) location for the upstream boundary

        Parameters
        ----------
        
        id:
            id (integer value) for boundary facets as defined on mesh

        location:
            location (m) of the upstream boundary (the mesh needs to be aligned parallel to x-axis, positive axis direction is from upstream to downstream boundary)
        """
    
        self.__upstream_boundary_id=id
        self.__upstream_boundary_location=location

    def set_downstream_boundary(self,id,location=1.0):
        """
        Sets the boundary (material) id  and the (x-axis) location for the downstream boundary

        Parameters
        ----------
        
        id:
            id (integer value) for boundary facets as defined on mesh
        
        location:
            location (m) of the downstream boundary (the mesh needs to be aligned parallel to x-axis, positive axis direction is from upstream to downstream boundary) 
        """
    
        self.__downstream_boundary_id=id
        self.__downstream_boundary_location=location

    
    def set_upstream_reservoir_concentration(self,conc):
        """
        Sets the concentration of the upstream reservoir

        Parameters
        ----------
        
        conc:
            concentration in mol/m³
        """
    
        self.__upstream_concentration=conc
        
    def get_upstream_reservoir_concentration(self):
        """
        returns the concentration of the upstream reservoir

        Returns
        -------
        
        conc:
            concentration in mol/m³
        """
    
        return self.__upstream_concentration    
        
    def set_upstream_reservoir_volume(self,volume):
        """
        Sets the volume of the upstream reservoir

        Parameters
        ----------
        
        volume:
            volume in m³
        """
    
        self.__upstream_volume=volume   
        
    def set_downstream_reservoir_concentration(self,conc):
        """
        Sets the concentration of the downstream reservoir

        Parameters
        ----------
        
        conc:
            concentration in mol/m³
        """
    
        self.__downstream_concentration=conc
    def get_downstream_reservoir_concentration(self):
        """
        returns the concentration of the downstream reservoir

        Returns
        -------
        
        conc:
            concentration in mol/m³
        """
    
        return self.__downstream_concentration
        
    def set_downstream_reservoir_volume(self,volume):
        """
        Sets the volume of the downstream reservoir

        Parameters
        ----------
        
        volume:
            volume in m³
        """
    
        self.__downstream_volume=volume  

    def set_downstream_reservoir_exchange_solution_times(self,time_array,rflag=True):
        """
        Provides a list of times at which the downstream concentrations were measured. Time stepping is controlled such that solutions are calculated for each time in the list.
        If set_update_reservoir_boundaries(True) the concentration of the downstream reservoir to zero (C=0 mol/m^3) at specified times 
        to mimic exchange of solution.

        Parameters
        ----------
        
        time_array:
            time vector in s

        rflag:
            optional boolean flag (default False) that indicates if concentration is exchanged at each time (default for downstream reservoir during in- and out-diffusion tests)
        """
    
        self.__downstream_exchange_times=np.array(time_array)
        self.__downstream_exchange_times_done=np.full(len(time_array), False)
                # reset up- or down-stream reservoir concentrations 
        self.__flag_exchange_downstream_reservoir=rflag # False works for in and out-diffusion   
        logging.debug('DEBUG: self.__downstream_exchange_times_done'+str(self.__downstream_exchange_times_done))
        
    def set_upstream_reservoir_exchange_solution_times(self,time_array,rflag=False):
        """
        Provides a list of times at which the upstream concentrations were measured. Time stepping is controlled such that solutions are calculated for each time in the list.
        If set_update_reservoir_boundaries(True) the concentration of the upstream reservoir to zero (C=0 mol/m^3) at specified times 
        to mimic exchange of solution.

        Parameters
        ----------
        
        time_array:
            time vector in s
            
        rflag:
            optional boolean flag (default False) that indicates if concentration is exchanged at each time (set to True for out-diffusion tests)             
        """
    
        self.__upstream_exchange_times=np.array(time_array)
        self.__upstream_exchange_times_done=np.full(len(time_array), False)
                # reset up- or down-stream reservoir concentrations 
        self.__flag_exchange_upstream_reservoir=rflag  # set to True for out-diffusion
        logging.debug('DEBUG: self.__upstream_exchange_times_done'+str(self.__upstream_exchange_times_done))              
        
    def downstream_exchange(self,t):
        """
        tests if a downstream exchange cycle should happen for given time t (+- 30 s), sets the downstream concentration to zero
        and returns (and sets) flag for yes (exchanged happened) or no
        
        Please note that time stepping need to be designed such that all exchanges times are met.

        Parameters
        ----------

        t:
        current simulation time

        Returns:
        --------
        rflag:
        bool flag which is True if exchange cycle is met
        """
        rflag=False
        if self.__flag_exchange_downstream_reservoir:
            for idx in range(len(self.__downstream_exchange_times_done)):
                if (not self.__downstream_exchange_times_done[idx]):
                    logging.debug('DEBUG: downstream_exchange: i, t, delta(t)'+str(idx)+str(t)+str(self.__downstream_exchange_times[idx]))
                    if (t > self.__downstream_exchange_times[idx]):
                        self.__downstream_exchange_times_done[idx]=True
                        self.set_downstream_reservoir_concentration(PETSc.RealType(0.0))
                        logging.debug('DEBUG: downstream_exchange time mismatch: i, t, t_exchange'+str(idx)+str(t)+str(self.__downstream_exchange_times[idx]))
                        rflag=True
                        break # do not continue loop ...
                    if np.isclose(self.__downstream_exchange_times[idx],t,atol=30.0): # everything within half a minute is accepted
                        self.__downstream_exchange_times_done[idx]=True
                        self.set_downstream_reservoir_concentration(PETSc.RealType(0.0))
                        logging.debug('DEBUG: downstream_exchange time hit: i, t, t_exchange'+str(idx)+str(t)+str(self.__downstream_exchange_times[idx]))
                        rflag=True
                        break # do not continue loop ...
        return rflag

    def upstream_exchange(self,t):
        """
        tests if a upstream exchange cycle should happen for given time t (+- 30 s), sets the upstream concentration to zero
        and returns (and sets) flag for yes (exchanged happened) or no
        
        Please note that time stepping need to be designed such that all exchanges times are met.

        Parameters
        ----------

        t:
        current simulation time

        Returns:
        --------
        rflag:
        bool flag which is True if exchange cycle is met
        """
        rflag=False
        if self.__flag_exchange_upstream_reservoir:
            for idx in range(len(self.__upstream_exchange_times_done)):
                if (not self.__upstream_exchange_times_done[idx]):
                    logging.debug('DEBUG: upstream_exchange: i, t, delta(t)'+str(idx)+str(t)+str(self.__upstream_exchange_times[idx]))
                    if (t > self.__upstream_exchange_times[idx]):
                        self.__upstream_exchange_times_done[idx]=True
                        self.set_upstream_reservoir_concentration(PETSc.RealType(0.0))
                        logging.debug('DEBUG: upstream_exchange time mismatch: i, t, t_exchange'+str(idx)+str(t)+str(self.__upstream_exchange_times[idx]))
                        rflag=True
                        break # do not continue loop ...
                    if np.isclose(self.__upstream_exchange_times[idx],t,atol=30.0): # everything within half a minute is accepted
                        self.__upstream_exchange_times_done[idx]=True
                        self.set_upstream_reservoir_concentration(PETSc.RealType(0.0))
                        logging.debug('DEBUG: upstream_exchange time hit: i, t, t_exchange'+str(idx)+str(t)+str(self.__upstream_exchange_times[idx]))
                        rflag=True
                        break # do not continue loop ...
        return rflag


        
    def check_downstream_exchange_dt(self,t,dt):
        """
        Tests if a downstream exchange cycle should happen for given time intervall (t,t+dt).
        if yes, an adjusted max(dt) is returned such that next exchange cycle is met 'exactly'.
        Otherwise the unchanged dt value is returned.
    

        Parameters
        ----------

        t:
        current simulation time

        dt:
        current time step size

        Returns:
        --------
        dt:
        time step size
        """
        _new_dt=dt
        for idx in range(len(self.__downstream_exchange_times_done)):
            if (not self.__downstream_exchange_times_done[idx]):
                if ( t < self.__downstream_exchange_times[idx]):
                    _new_dt=self.__downstream_exchange_times[idx]-t
                    logging.debug('DEBUG: check_downstream_exchange_dt: i,t, next t, dt,_new_dt:'+str(idx)+" , "+str(t)+
                                 " , "+str(self.__downstream_exchange_times_done[idx])+" ,"+str(dt)+" , "+str(_new_dt))
                    break
        if _new_dt < 0.0:
            logging.warning("WARNING: Something went wrong during check_downstream_exchange_dt. time step not changed. dt, _new_dt: "+str(dt)," ",+str(_new_dt))
            _new_dt=dt

        return _new_dt

    def check_upstream_exchange_dt(self,t,dt):
        """
        Tests if a upstream exchange cycle should happen for given time intervall (t,t+dt).
        if yes, an adjusted max(dt) is returned such that next exchange cycle is met 'exactly'.
        Otherwise the unchanged dt value is returned.
    

        Parameters
        ----------

        t:
        current simulation time

        dt:
        current time step size

        Returns:
        --------
        dt:
        time step size
        """
        _new_dt=dt
        for idx in range(len(self.__upstream_exchange_times_done)):
            if (not self.__upstream_exchange_times_done[idx]):
                if ( t < self.__upstream_exchange_times[idx]):
                    _new_dt=self.__upstream_exchange_times[idx]-t
                    logging.debug('DEBUG: check_upstream_exchange_dt: i,t, next t, dt,_new_dt:'+str(idx)+" , "+str(t)+
                                 " , "+str(self.__upstream_exchange_times_done[idx])+" ,"+str(dt)+" , "+str(_new_dt))
                    break
        if _new_dt < 0.0:
            logging.warning("WARNING: Something went wrong during check_upstream_exchange_dt. time step not changed. dt, _new_dt: "+str(dt)," ",+str(_new_dt))
            _new_dt=dt

        return _new_dt

        
    def set_cross_section_area(self,area):
        """
        Sets the cross section area for the diffusion sample

        Parameters
        ----------
        
        area:
            area in m²
        """
    
        self.__cross_section_area=area    

    # --- New methods for Temperature feedback (English comments added) ---
    def set_temperature(self, Temp_K: float):
        """ set current temperature field (Unit: K) """
        self.__Temp.x.array[:] = PETSc.RealType(Temp_K)

    def update_De_from_Temp_arrhenius(self, De_ref_dict, Ea_Jmol, Temp_ref_K=None):
        """ Update effective diffusion coefficient De based on current temperature field using Arrhenius law """
        R = 8.314462618
        if Temp_ref_K is None:
            Temp_ref_K = float(self.__Temp_ref)
    
        # Project CG1 Temperature to DG0 space for cell-wise material update
        self.__Temp0.interpolate(self.__Temp)
    
        for imat, De_ref in De_ref_dict.items():
            cells = self.cell_tag.find(imat)
            Temp_cells = self.__Temp0.x.array[cells]
    
            De_cells = De_ref * np.exp(
                -(Ea_Jmol / R) * (1.0 / Temp_cells - 1.0 / Temp_ref_K)
            )
    
            self.__De.x.array[cells] = De_cells.astype(PETSc.RealType)
        
    def set_darcy_flux(self,darcy_flux_vector):
        """
        Defines a advective flux (water flow) constant in the domain. 
        
        Parameters
        ----------
        
        darcy_flux_vector:
            A 2D vector with darcy flux components in 0,1 direction is required for a 2D mesh, a vector with 3 components is required for a 3D mesh.


        
        """
        def darcy_flux_init(x):
            """
            function for use with FEniCS for interpolation of flux vector onto FE spaces

            Parameters
            ----------
        
            x:
            coordinate Object , automatically supplied by FEniCSx

            Returns
            -------
            value at given coordinate
            
            """
            values = np.zeros((self.mymesh.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType)
            if self.mymesh.geometry.dim == 3 :
                values[0] = darcy_flux_vector[0]
                values[1] = darcy_flux_vector[1]
                values[2] = darcy_flux_vector[2]
            elif self.mymesh.geometry.dim == 2 :
                values[0] = darcy_flux_vector[0]
                values[1] = darcy_flux_vector[1]
            else :
                logging.critical("CRITICAL ERROR in set_darcy_flux: problem with Darcy flux vector definition")    
                sys.exit()  # Terminate the program with status code 0
            return values

        if len(darcy_flux_vector) == self.mymesh.geometry.dim :
            logging.info("Darcy flux is set to: "+str(darcy_flux_vector)+" for the whole domain.")
            self.__darcy_flux.interpolate(darcy_flux_init) # this is the actual interpolation of darcy flux
        else:
            logging.critical("CRITICAL set_darcy_flux: Exiting code: Something is wrong with the darcy flux vector: Expected length: "+str(self.mymesh.geometry.dim)+ " I got something of length: "+str(len(darcy_flux_vector)))
            sys.exit()
        
    def set_initial_conditions(self, conc0=0.0):
        """
        function for defining initial tracer concentrations (constant in domain) 

        Parameters
        ----------
        
        conc0:
            initial concentration in the domain
            
        """
        self.u0.x.array[:]=PETSc.RealType(conc0)
        self.uh.interpolate(self.u0)   # set the same to uh

        return self.u0

    def set_initial_conditions_out_diffusion(self, concu=1.0, concl=0.0, xu=0.0,xl=1.0):
        """
        function for defining initial tracer concentrations in 1D (linear interpolation beween concu and concl) 

        Parameters
        ----------
        
        concu:
            initial concentration at the upstream boundary (x=xu)

        concl:
            initial concentration at the downstream boundary (x=xl)
        xu:
            position of upstream boundary 
        xl:
            position of downstream boundary 
            
        """

        # linear interpolation y=cu
        self.u0.interpolate(lambda x: concu+(x[0]-xu)*(concu-concl)/(xu-xl))
        self.uh.interpolate(self.u0)   # set the same to uh

        return self.u0
        
    def set_boundary_conditions(self, boundary_conditions):
        """
        function for defining a list that contains boundary conditions 

        Parameters
        ----------
        
        boundary_conditions:
            list with definitions for boundary conditons
            
        """
        self.boundary_conditions=boundary_conditions
        logging.debug("Debug set boundary conditions: "+str(self.boundary_conditions))
        
    def set_DeT(self):
        """
        sets anisotropy sensor to unit tensor
        """
        def DeT_tensor(x):
            """
            function for use with FEniCS fem.interpolate() for interpolation of a anisotropy Tensor onto FE spaces
            the resulting diffusion tensor is: DeT*De
            
            In this version the unit-tensor is hardcoded, i.e. diffusion will be always isotropic.

            Parameters
            ----------
        
            x:
            coordinate Object , automatically supplied by FEniCSx
            
            Returns
            -------
            
            tensor at given coordinate
            
            """
            tensor = np.zeros((self.mymesh.geometry.dim*self.mymesh.geometry.dim,
                      x.shape[1]), dtype=np.float64)
            if self.mymesh.geometry.dim == 3 :
                #print("DeT for dimension ", self.mymesh.geometry.dim)
                tensor[0,:] = 1
                tensor[4,:] = 1 # check position
                tensor[8,:] = 1 # check position
            elif self.mymesh.geometry.dim == 2 :
                #print("DeT for dimension ", self.mymesh.geometry.dim)
                tensor[0,:] = 1
                tensor[3,:] = 1
            else :
                logging.critical("CRITICAL in set_DeT: Problem with mesh? Tensor not for 2D or 3D ? tensor: "+str(tensor))
                sys.exit()
            return tensor
        self.__DeT.interpolate(DeT_tensor) # function call to DeT_tensor()
        
    def set_De(self, De_dict={0:1.0e-11}):
        """
        sets De values for different material groups

        Parameters
        ----------
        
        De_dict:
        Dictionary that contains the material group (as defined for the mesh) and the associated effective diffusion coefficient,
        i.e. {11:1.0e-11, 12:2.0e-12} for material 11 a De of 1.0e-11 is defined, for material 12 a De of 2e-12 is defined.
        
        The default De of 1.0e-11 is assigned to all materials (cells) that are not included in the dictionary.

  
        
        """
        if isinstance(De_dict, dict):
            logging.info("INFO: De set via dictionary "+str(De_dict)+" default: 1.0e-11 ")
        else:
            logging.critical("CRITICAL in set_De: De is not a dictionary. I got \""+str(De_dict)+"\" Exiting \n")
            sys.exit()
        # set default value
        self.__De.x.array[:]=PETSc.RealType(1.0e-11) # setting default to zero will introduce a convergence problem...
        for imat,value in De_dict.items():
            mat_cells=self.cell_tag.find(imat)
            self.__De.x.array[mat_cells] = np.full_like(mat_cells, value, dtype=PETSc.RealType)
        

    def set_porosity(self, porosity_dict={0 : 1.0}): # version with dictionary
        """
        sets De values for different material groups

        Parameters
        ----------
        
        porosity_dict:
        Dictionary that contains the material group (as defined for the mesh) and the associated porosity

        The default porosity of 1 is assigned to all materials (cells) that are not included in the dictionary.
        """
        if isinstance(porosity_dict, dict):
            logging.info("porosity set via dictionary "+str(porosity_dict)+" default: 1.0")
        else:
            logging.critical("CRITICAL in set_porosity: porosity is not a dictionary. I got \""+str(De_dict)+"\" Exiting \n")
            sys.exit()
        
        # set default value 
        self.__porosity.x.array[:]=PETSc.RealType(1.0)                    
        for imat,value in porosity_dict.items():
            mat_cells=self.cell_tag.find(imat)
            self.__porosity.x.array[mat_cells] = np.full_like(mat_cells, value, dtype=PETSc.RealType)

#########################    flux calculations and related functions for boundary reservoirs #########################        
        
    def calculate_diffusive_fluxes(self,uh,id):   #mesh, solution vector, id of boundary
        """
        Calculates specific solute fluxes across a specific boundary  
        The tracer flow is normalized with respect of boundary area.
        
        Parameters:
        -----------
        uh:
        solution (concentrations) for which fluxes should be calculated

        id:
        mesh tags for identification of the faces that form the boundary

        Returns:
        --------
        flux:
        flux across boundary (mol s⁻² m⁻²)

        mesh_area:
        cross sectional area of boundary from mesh

        """
        #n = FacetNormal(self.mymesh)
        #ds = Measure("ds", domain=self.mymesh,subdomain_data=facet_tag) # ds(1) is inflow boundary ...ds(2) is outflow boundary
        n = FacetNormal(self.mymesh)
        ds = Measure("ds", domain=self.mymesh,subdomain_data=self.facet_tag)

        dummy=(fem.form((1.0)*ds(id))) # length/area for the boundary
        _boundary_area=self.mymesh.comm.allreduce(fem.assemble_scalar(dummy), op=MPI.SUM)
        if (_boundary_area < 1.0e-16) :
            logging.error("ERROR: boundary area: ",_boundary_area, " boundaries defined correctly?")
        
        outflux_correct=(fem.form(dot(n,self.__De*self.__DeT*grad(uh))*ds(id)))
        flux_correct=self.mymesh.comm.allreduce(fem.assemble_scalar(outflux_correct), op=MPI.SUM)
        
        return flux_correct/_boundary_area, _boundary_area
        
    def update_upstream_concentration(self,uh,dt):   #mesh, solution vector, id of boundary
        """
        Calculates solute fluxes across upstream boundary and changes concentration for upstream reservoir

        Parameters
        ----------
        
        uh:
        solution vector

        dt:
        time step size

        Returns
        -------
        Concentration in upstream reservoir

        """
        #n = FacetNormal(self.mymesh)
        #ds = Measure("ds", domain=self.mymesh,subdomain_data=facet_tag) # ds(1) is inflow boundary ...ds(2) is outflow boundary
        n = FacetNormal(self.mymesh)
        ds = Measure("ds", domain=self.mymesh,subdomain_data=self.facet_tag)
        
        dummy=(fem.form((1.0)*ds(self.__upstream_boundary_id))) # length/area for the boundary
        _mesh_area=self.mymesh.comm.allreduce(fem.assemble_scalar(dummy), op=MPI.SUM)
        outflux_correct=(fem.form(dot(n,self.__De*self.__DeT*grad(uh))*ds(self.__upstream_boundary_id)))
        flux_correct=self.mymesh.comm.allreduce(fem.assemble_scalar(outflux_correct), op=MPI.SUM)
         # for upstream reservoir
        _C_change=dt*flux_correct/_mesh_area*self.__cross_section_area/self.__upstream_volume
        # this is sensitive to meshing and coordinate system and position of boundary....
        # positive fluxes are into model domain, negative ones are out of the domain

        self.__upstream_concentration=self.__upstream_concentration-_C_change 
        
        return self.__upstream_concentration     

    def update_downstream_concentration(self,uh,dt):   #mesh, solution vector, id of boundary
        """
        Calculates solute fluxes across upstream boundary and changes concentration for downstream reservoir  
        
        Parameters
        ----------
        
        uh:
        solution vector

        dt:
        time step size

        Returns
        -------
        Concentration in downstream reservoir

        """
        #n = FacetNormal(self.mymesh)
        #ds = Measure("ds", domain=self.mymesh,subdomain_data=facet_tag) # ds(1) is inflow boundary ...ds(2) is outflow boundary
        n = FacetNormal(self.mymesh)
        ds = Measure("ds", domain=self.mymesh,subdomain_data=self.facet_tag)
        
        dummy=(fem.form((1.0)*ds(self.__downstream_boundary_id)))
        _mesh_area=self.mymesh.comm.allreduce(fem.assemble_scalar(dummy), op=MPI.SUM)
        outflux_correct=(fem.form(dot(n,self.__De*self.__DeT*grad(uh))*ds(self.__downstream_boundary_id)))
        flux_correct=self.mymesh.comm.allreduce(fem.assemble_scalar(outflux_correct), op=MPI.SUM)
         # for upstream reservoir
        _C_change=dt*flux_correct/_mesh_area*self.__cross_section_area/self.__downstream_volume
        # this is sensitive to meshing and coordinate system and position of boundary....
        # positive fluxes are into model domain, negative ones are out of the domain

        self.__downstream_concentration=self.__downstream_concentration-_C_change 
        
        return self.__downstream_concentration


###################################    write to files for postprocessing #########################
    def set_vtx_filename(self,filename, i):
        """sets name for output of vtx files every i timesteps 

        Parameters
        ----------
        
        filename:
            string with (relative or absolute) path + filename
        i:
            intervall for output of results ( >=1)
        """
        self.__vtx_filename = filename
        self.__vtx_intervall = int(i)
        
    def vtx_write_materials(self,filename_vtx,t=0.0) :
        """ 
        write material properties with vtx writer (Read in latest parallel version of Paraview with ADIOS2 reader)

        Parameters
        ----------
        
        filename_vtx:
        Name of file with ending .bp
    
        t:
        time for which this is written is required
        """
        self.vtx_mat = io.VTXWriter(self.mymesh.comm, filename_vtx, (self.__De,self.__porosity,self.__DeT))
        self.vtx_mat.write(t)
        self.vtx_mat.close()

    def vtx_write_timestep(self,filename_vtx,solution,t) :
        """ 
        write result of diffusion calculation into VTX  writer (Read in latest parallel version of Paraview with ADIOS2 reader)

        Parameters
        ----------
        
        filename_vtx:
        Name of file with ending .bp
        
        solution:
        solution that should be written to file
    
        t:
        time for which this is written is required
        """
        try: 
            self.vtx
        except AttributeError:
            self.vtx = io.VTXWriter(self.mymesh.comm, filename_vtx, solution)
        else:
            self.vtx.write(t)

    
    def xdmf_write_timestep(self,filename_vtk,solution,t) :
        """ 
        opens and write initial condition and material properties with vtk writer 
        Works only for lagrange elements (results from discontinuous Galerkin element space have to be interpolated onto Lagrange element space)
        If the initial conditions were already written, only the results are written for the new time step.

        Parameters
        ----------
        
        filename_vtk:
        Name of file with ending .xdmf
        
        solution:
        solution that should be written to file, will be interpolated onto lagrange elements
    
        t:
        time for which this is written is required
        """
        try: 
            self.xdmf
        except AttributeError:    
            self.xdmf = io.VTKFile(self.mymesh.comm, filename_vtk, "w")
            #xdmf.write_mesh(self.mymesh)
            self.xdmf.write_function(self.__De)
            self.xdmf.write_function(self.__porosity)
            self.xdmf.write_function(self.__DeT)
            self.__uout.interpolate(solution) # interpolate
            self.xdmf.write_function(self.__uout, t) # initial state
        else:
            # should give an error if xdmf does not exist
            self.__uout.interpolate(solution) # interpolate to simple Lagrange elements
            self.xdmf.write_function(self.__uout, t) # write 

    def xdmf_write_close(self):
        """
        Properly closes xdmf file and quits writer 
        """
        self.xdmf.close()


        
################################### ADE core solver #################################        
    def ADEsolver(self,u_old,u_new,boundary_conditions,dt_val=1.0,mytheta=0.5):
        """
        defines the weak form of the ADE in ufl language from which the FE equation solver is constructed

        Parameters
        ----------

        u_old:
        data for t(n), current timestep

        u_new:
        data for t(n+1), next time step
        
        boundary_conditions:
        list object with boundary condition definitions (see FENICSx Tutorial on multiple boundary definitions)
        
        dt_val:
        time step size 

        mytheta:
        theta factor for implicit (theta 0.5, 1.0) /explicit (theta=0.0) 

        Returns:
        --------
        solution vector
        
        """

        # set time step
        dt = fem.Constant(self.mymesh, PETSc.RealType(dt_val))
        # Extract function space
        #V = u_new.function_space()
        #V=self.V  # if u_new is not made from V or V is not DG everything blows up

       # Define unknown and test function(s)
        v = TestFunction(self.V)
        u = TrialFunction(self.V)
#        # Define solution variable
#        self.uh = fem.Function(self.V)
#        self.uh.name = "uh"
#        self.uh.interpolate(self.u0)
        
        #  Source term.
        x = SpatialCoordinate(self.mymesh)

        f = fem.Constant(self.mymesh, PETSc.RealType(0.0)) # no source-sink 
        
        dx = Measure("dx", domain=self.mymesh, subdomain_data=self.cell_tag)
        ds = Measure("ds", domain=self.mymesh,subdomain_data=self.facet_tag)
        dS = Measure("dS", domain=self.mymesh)
        

         
        # time discretization ..either theta =0.5 or 1.0 (implicit)
        #theta = fem.Constant(self.mymesh, PETSc.ScalarType(0.5)) # I guess it should also work like this
        theta=PETSc.RealType(mytheta) # implicit is stable, best works for Crank-Nicolson ...fully implicit will give deviation in front shape (numerical diffusion!)
        dt = fem.Constant(self.mymesh, dt_val)        
        # STABILIZATION
        n = FacetNormal(self.mymesh)
        alpha = fem.Constant(self.mymesh, PETSc.ScalarType(3.0))       
        vc = CellVolume(self.mymesh)
        fc = FacetArea(self.mymesh)
        #h_avg = 0.5*(h('+') + h('-'))
        h_avg = (vc('+') + vc('-'))/(2*avg(fc))
        # normal water flux over face
        bn = (dot(self.__darcy_flux, n) + abs(dot(self.__darcy_flux, n)))/2.0
        
        # following implementation of Kadeethum 2021
        def avg_w(x,w):
            return (w*x('+')+(1-w)*x('-'))
        
        def k_normal(k,n):
            return dot(dot(np.transpose(n),k),n)
        
        def k_plus(k,n):
            return dot(dot(n('+'),k('+')),n('+'))
        
        def k_minus(k,n):
            return dot(dot(n('-'),k('-')),n('-'))
        
        def weight_e(k,n):
            return (k_minus(k,n))/(k_plus(k,n)+k_minus(k,n)+1.0e-20)
        
        def k_e(k,n):
            return (2*k_plus(k,n)*k_minus(k,n)/(k_plus(k,n)+k_minus(k,n)+1.0e-20))
        
        gamma_int = alpha/h_avg *k_e(self.__De*self.__DeT,n) * (self.__deg*(self.__deg+self.fdim-1))
        
        
        def aa(u,v) :
            # Bilinear form
            a_int = dot(grad(v), (self.__De*self.__DeT)*grad(u) - self.__darcy_flux*u)*dx
            
            a_fac =  gamma_int*dot(jump(u, n), jump(v, n))*dS \
                    - dot(avg_w(self.__De*self.__DeT *grad(u),weight_e(self.__De*self.__DeT,n)) , jump(v, n))*dS \
                    - dot(jump(u, n), avg_w(self.__De*self.__DeT *grad(v),weight_e(self.__De*self.__DeT,n)) )*dS
        
            a_vel = dot(jump(v), (bn('+')*u('+')) - (bn('-')*u('-') ))*dS  + dot(v, bn*u)*ds # has probably to be adjusted according to boundary fluxes
            
            a = a_int + a_fac + a_vel
            return a
        
        a0=aa(u_old,v)
        a1=aa(u,v)
        #
        F = dt*theta*a1 + dt*(1-theta)*a0 - self.__porosity*inner(u_old,v)*dx + self.__porosity*inner(u, v)*dx - dt*v*f*dx

        # disable decay for the moment
        Half_life =0.0
        if (Half_life > 0.0):
            # decay term
            k=np.log(2.0)/Half_life
            l = fem.Constant(self.mymesh,PETSc.RealType(k))
            F += dt*l*inner(u_old,v)*dx   # the decay term is based on last time step
        
        # this is how it it done in https://jsdokken.com/dolfinx-tutorial/chapter3/robin_neumann_dirichlet.html
        mybcs = []
        for condition in boundary_conditions:
            if condition.type == "Dirichlet":
                mybcs.append(condition.bc)
                #print("Debug: Dirichlet boundary set")
            else:
                F += condition.bc
        
        a = lhs(F)
        L = rhs(F)
        
        #problem = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        
        a_compiled = fem.form(a)
        L_compiled = fem.form(L)
        
       # Assemble matrix which does not change
        A = assemble_matrix(a_compiled, bcs=mybcs)
        A.assemble()
        b = create_vector(L_compiled)
        
        ######################33 create solver ...dirct solver LU
#        self.solver = PETSc.KSP().create(self.mymesh.comm)
#        self.solver.setOperators(self.A)
#        self.solver.setType(PETSc.KSP.Type.PREONLY)
#        self.solver.getPC().setType(PETSc.PC.Type.LU)
#        jit_options={"timeout": 60}
        _solver = PETSc.KSP().create(self.mymesh.comm)
        _solver.setOperators(A)
        _solver.setType(PETSc.KSP.Type.PREONLY)
        _solver.getPC().setType(PETSc.PC.Type.LU)

        def solve_(boundary_conditions,tstep,A_flag):
            """
            solves the PDE for one time step and returns solution uh ( = fem.Function(V))
    
            boundary_conditions:
            list object with boundary condition definitions (see FENICSx Tutorial on multiple boundary definitions)
            
            tstep:
            time step size (seconds) 
    
            A_flag:
            boolean (True or False) to indicate that re-assemply of the A matrix is required. This is necessary if e.g. the time step was changed.
    
    
            """
    
    # for changing concentrations we need to update Dirichlet boundary condition for each time step...this is probably the slow way
            mybcs = []
            for condition in boundary_conditions:
                if condition.type == "Dirichlet":
                    mybcs.append(condition.bc)
                    logging.debug("DEBUG: Dirichlet boundary set ")
    
    # re assemble matrix not always needed, 
            # needed if e.g. dt or porosity, De is changing
            if  A_flag :   
                dt.value=tstep # update a time step ...will be only done, if A_flag is set......
                A = assemble_matrix(a_compiled, bcs=mybcs)
                A.assemble()
                _solver.setOperators(A)
            
            # Update the right hand side reusing the initial vector
            with b.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b, L_compiled)
            
            fem.apply_lifting(b, [a_compiled], [mybcs])
    #        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            # Apply Dirichlet boundary condition to the vector
            fem.set_bc(b, mybcs)
        
            # Solve linear problem
            _solver.solve(b, u_new.x.petsc_vec)
    #        u_new.x.scatter_forward()
           
            return u_new
            
        return solve_  
        
        
#########################   fully automated time stepping #############################
        
    def solve_adaptive_timesteps(self, Temp_input=None):
        """Perform adaptive timestepping using theta-scheme with Temperature support. 
        """
        def compute_est(theta, u_L, u_H):
            """Return error estimate by Richardson extrapolation"""
            p = 2 if theta == 0.5 else 1
            est = np.sqrt(fem.assemble_scalar(fem.form((u_L - u_H)**2*dx)) / (2**p - 1))
            return est
        
        
        def compute_new_dt(theta, est, tol, dt):
            """Return new time step"""
            p = 2 if theta == 0.5 else 1
            rho = 0.9
            dt_new = dt * ( rho * tol / (est + 1e-20) )**(1/p)
            return dt_new  

        # set boundary conditions specific for diffusion tests
        _x_ubc=self.__upstream_boundary_location # left boundary, where the upstream concentrations are set
        _x_dbc=self.__downstream_boundary_location# right boundary, where the downstream concentrations are set
        up_bc=self.__upstream_concentration
        down_bc=self.__downstream_concentration
        # as topological search does not work for DG, functions for geometric search are needed
        def left(x):
            return np.isclose(x[0], _x_ubc)
        
        def right(x):
            return np.isclose(x[0], _x_dbc)
        
        boundary_conditions = [BoundaryCondition(self,'Dirichlet', left, up_bc),
                            BoundaryCondition(self,'Dirichlet', right, down_bc)]
        
        # Initialize needed functions
        u_n = fem.Function(self.V)
        u_np1_low = fem.Function(self.V)
        u_np1_high = fem.Function(self.V)    

        # needed for energy calculations
        dx = Measure("dx", domain=self.mymesh, subdomain_data=self.cell_tag)
        
        # Initial time step; the value does not really matter
        T=self.__T_end
        dt = self.__dt
        tol=self.__tol_adaptive
        theta=self.__theta
        dtmax=T/10.0  # just to set something which is not too small!
        
        # Prepare solvers for computing tentative time steps
        solver_low = self.ADEsolver(u_n, u_np1_low,boundary_conditions,dt,theta)
        solver_high_1 = self.ADEsolver(u_n, u_np1_high,boundary_conditions,dt/2.0,theta)
        solver_high_2 = self.ADEsolver(u_np1_high, u_np1_high,boundary_conditions,dt/2.0,theta)

        # pandas dataframe for recording results
        parameters = ['time', 'dt', 'Upstream_concentration', 'Downstream_concentration', 'Upstream_flux', 'Downstream_flux']
        df = pd.DataFrame(columns=parameters)
    
        # Set initial conditions
        u_n.interpolate(self.u0)
        #
        logging.info("INFO: Adaptive time loop starts with error tolerance: "+str(tol)+" update_reservoir_concentrations: "+str(self.__flag_reservoirs))
        #
            # Perform timestepping
        t = self.__T_start
        
        icount=0
        while t < T :
            # Report some numbers
            logging.debug("DEBUG: t dt error est {:10.4f} | {:10.4f} | {:#10.4g}".format(t, dt, fem.assemble_scalar(fem.form(u_n*dx))))

            # --- Handle Temperature Updates (New Logic) ---
            A_flag = False
            if Temp_input is not None:
                # Get current temperature value
                if isinstance(Temp_input, (float, int)): Temp_eval = float(Temp_input)
                elif callable(Temp_input): Temp_eval = float(Temp_input(t))
                else: Temp_eval = float(Temp_input[min(icount, len(Temp_input)-1)])

                # Detect if temperature change is significant
                if (self.__Temp_last is None) or (abs(Temp_eval - self.__Temp_last) > 1e-8):
                    self.set_temperature(Temp_eval)
                    # Update De field (Using default reference dictionary as example)
                    self.update_De_from_Temp_arrhenius({0: 1.0e-11}, 25000.0) 
                    A_flag = True # Force matrix re-assembly
                    self.__Temp_last = Temp_eval

            # --- Compute tentative time steps using Richardson Extrapolation ---
            solver_low(boundary_conditions, dt, A_flag)         # Big step
            solver_high_1(boundary_conditions, dt/2.0, True)    # Small step 1 (Force reassemble due to dt/2)
            solver_high_2(boundary_conditions, dt/2.0, True)    # Small step 2
    
            # Compute error estimate and new timestep
            est = compute_est(theta, u_np1_low, u_np1_high)
            dt_new = compute_new_dt(theta, est, tol, dt)
            
            if est > tol:
                # Tolerance not met; repeat the step with new timestep
                dt = dt_new
                logging.debug('DEBUG during time iteration: retry step at t: '+str(t))
                continue

            # --- Move to next time step (Accept step) ---
            logging.info("INFO: t dt error est {:10.4f} | {:10.4f} | {:#10.4g}".format(t, dt, est))  
            
            # calculate fluxes across boundaries
            dflux, _ = self.calculate_diffusive_fluxes(u_np1_high, self.__downstream_boundary_id)
            uflux, _ = self.calculate_diffusive_fluxes(u_np1_high, self.__upstream_boundary_id)
             
            if self.__flag_reservoirs: 
                up_bc=self.update_upstream_concentration(u_np1_high,dt)
                down_bc=self.update_downstream_concentration(u_np1_high,dt)
            else:
                up_bc, down_bc = self.__upstream_concentration, self.__downstream_concentration
                
            new_row = pd.DataFrame([[t,dt,up_bc,down_bc,uflux,dflux]], columns=parameters)
            df = pd.concat([df, new_row], ignore_index=True)
            
            # VTX file output
            if (self.__vtx_intervall >= 1) and (icount % self.__vtx_intervall == 0):
                self.vtx_write_timestep(self.__vtx_filename, u_np1_high, t)
                    
            # Prepare data for next iteration
            u_n.x.array[:] = u_np1_high.x.array[:]
            t += dt
            icount +=1
            
            # Align dt with reservoir sampling times
            dtreservoir = self.check_downstream_exchange_dt(t, dt_new)
            dtreservoir_up = self.check_upstream_exchange_dt(t, dtreservoir)
            dt = min(dt_new, dtreservoir_up, T - t + 1e-10) # Safety clamp for end time
            
        logging.info("INFO: Time loop finished at t="+str(t))
        return u_n, df

##################################### solve time dependet problem ###################        
    def solve_timesteps_simple(self):
        """Solve system for constant time steps
        
        """

        #  define functions 
        u_h = fem.Function(self.V)
        u_0 = fem.Function(self.V)

        # set initial conditions
        u_0.x.array[:] = self.u0.x.array[:] # old time step
        u_h.x.array[:] = self.u0.x.array[:] # new time step
        
        # Initial time step
        T=self.__T_end
        dt = self.__dt
        tol=self.__tol_adaptive
        theta=self.__theta
        dtmax=self.__dt_max # just to set something!
        A_flag =True # it is save (and slower) to set it true, because dt is changing and re-assembly of A is required

        boundary_conditions=(self.boundary_conditions)
        
        # Prepare solver for computing tentative time steps
        solver = self.ADEsolver(u_0, u_h,boundary_conditions,dt,theta)

            # Perform timestepping
        t = self.__T_start
        while t < T:
            # Report some numbers
            energy = fem.assemble_scalar(fem.form(self.uh*dx))
            logging.info("INFO: t dt error est {:10.4f} | {:10.4f} | {:#10.4g}".format(t, dt, energy))
    
            # Compute tentative time steps
            solver(boundary_conditions,dt,A_flag) # 
    
            # Move to next time step
            u_0.x.array[:] = u_h.x.array[:]
            t += dt

        return u_h
        
######################solve time dependent problem with changes in upstream and downstream reservoir        
    def solve_timesteps(self):
        """Solve system for constant time steps and changes in reservoirs...
        attention: boundary reservoir properties need to be defined!


            Returns: solution, data

            solution:
                Fenics fem.Function(V) object
            data:
                pandas data-frame with a row for each time step and with 6 collumns 
                'time', 'dt', 'Upstream_concentration', 'Downstream_concentration', 'Upstream_flux', 'Downstream_flux'
        """
        
                
        _x_ubc=self.__upstream_boundary_location # left boundary, where the upstream concentrations are set
        _x_dbc=self.__downstream_boundary_location# right boundary, where the downstream concentrations are set
        up_bc=self.__upstream_concentration
        down_bc=self.__downstream_concentration
        # as topological search does not work for DG, functions for geometric search are needed
        def left(x):
            return np.isclose(x[0], _x_ubc)
        
        def right(x):
            return np.isclose(x[0], _x_dbc)
        
        boundary_conditions = [BoundaryCondition(self,'Dirichlet', left, up_bc),
                            BoundaryCondition(self,'Dirichlet', right, down_bc)]

        
        #  define functions 
        u_h = fem.Function(self.V)
        u_0 = fem.Function(self.V)

        # set initial conditions
        u_0.x.array[:] = self.u0.x.array[:] # old time step
        u_h.x.array[:] = self.u0.x.array[:] # new time step
        
        # Initial time step
        T=self.__T_end
        dt = self.__dt
        theta=self.__theta
        dtmax=self.__dt_max # just to set something!
        A_flag =True # it is save (and slower) to set it true, because dt is changing and re-assembly of A is required

        # one would like to return the last solution and a vector with values for each time step
        # the time step dependent data we put into an pandas dataframe
        parameters = ['time', 'dt', 'Upstream_concentration', 'Downstream_concentration', 'Upstream_flux', 'Downstream_flux']
        df = pd.DataFrame(columns=parameters)
        
        # Prepare solver for computing tentative time steps
        solver = self.ADEsolver(u_0, u_h,boundary_conditions,dt,theta)

        # Perform timestepping
        t = self.__T_start
        i=1 # counter for time steps
        while t < T:
#            boundary_conditions = [BoundaryCondition(self,'Dirichlet', left, up_bc),
#                                BoundaryCondition(self,'Dirichlet', right, down_bc)] # I do not know if this is really needed..it seems boundary condition values are update even without this commands, or?
            
            # Compute tentative time steps
            solver(boundary_conditions,dt,A_flag) # 
            # calculate some stuff and save in padas data frame
            dflux,_down_area=self.calculate_diffusive_fluxes(u_h,self.__downstream_boundary_id) # this we do for downstream only
            
            uflux,_up_area=self.calculate_diffusive_fluxes(u_h,self.__upstream_boundary_id) # this we do for upstream only
             
            if self.__flag_reservoirs == True:
                up_bc=self.update_upstream_concentration(u_h,dt)
                down_bc=self.update_downstream_concentration(u_h,dt)
                
            # no append for pandas dataframe anymore ...we have to use concat
            new_row = pd.DataFrame([[t,dt,up_bc,down_bc,uflux,dflux]], columns=parameters)
            df = pd.concat([df, new_row], ignore_index=True)
            
                # Save to file
            if (self.__vtx_intervall >= 1) :
                if (i%self.__vtx_intervall == 0):
                    # self.xdmf_write_timestep(filename_xdmf,u_h,t)
                    self.vtx_write_timestep(self.__vtx_filename,u_h,t)

            # print something, but not every step
            if i%10 == 0:
                logging.info("t dt u_conc d_conc "+str([t,dt,up_bc,down_bc,dflux]))

            # Move to next time step
            u_0.x.array[:] = u_h.x.array[:] # the current solution is old solution for next time step 
            t += dt  # increase time
            i += 1   # increase time step number


        return u_h, df