/**
 * @file ITCModel.hpp
 * @brief Implementation of the Boussinesq thermal convection in a sphere
 * (Toroidal/Poloidal formulation)
 */

#ifndef QUICC_MODEL_BOUSSINESQ_SPHERE_TC_ITCMODEL_HPP
#define QUICC_MODEL_BOUSSINESQ_SPHERE_TC_ITCMODEL_HPP

// System includes
//
#include <string>

// Project includes
//
#include "QuICC/Generator/StateGenerator.hpp"
#include "QuICC/Generator/VisualizationGenerator.hpp"
#include "QuICC/Model/IPhysicalPyModel.hpp"
#include "QuICC/Simulation/Simulation.hpp"

namespace QuICC {

namespace Model {

namespace Boussinesq {

namespace Sphere {

namespace TC {

/**
 * @brief Implementation of the Boussinesq thermal convection sphere model
 * (Toroidal/Poloidal formulation)
 */
class ITCModel : public IPhysicalPyModel<Simulation, StateGenerator,
                    VisualizationGenerator>
{
public:
   /**
    * @brief Constructor
    */
   ITCModel() = default;

   /**
    * @brief Destructor
    */
   virtual ~ITCModel() = default;

   /// Formulation used for vector fields
   virtual VectorFormulation::Id SchemeFormulation() override;

   /**
    * @brief Version string
    */
   std::string version() const final;

   /**
    * @brief Add the required equations
    *
    * @param spSim   Shared simulation object
    */
   virtual void addEquations(SharedSimulation spSim) override;

   /**
    * @brief Add the initial state generation equations
    *
    * @param spGen   Shared generator object
    */
   virtual void addStates(SharedStateGenerator spGen) override;

   /**
    * @brief Add the visualization generation equations
    *
    * @param spGen   Shared visualization generator
    */
   virtual void addVisualizers(SharedVisualizationGenerator spVis) override;

   /**
    * @brief Add the required ASCII output files
    *
    * @param spSim   Shared simulation object
    */
   virtual void addAsciiOutputFiles(SharedSimulation spSim) override;

   /**
    * @brief XML configuration tags
    */
   virtual std::map<std::string, std::map<std::string, int>>
   configTags() const override;

protected:
private:
};

} // namespace TC
} // namespace Sphere
} // namespace Boussinesq
} // namespace Model
} // namespace QuICC

#endif // QUICC_MODEL_BOUSSINESQ_SPHERE_TC_ITCMODEL_HPP
