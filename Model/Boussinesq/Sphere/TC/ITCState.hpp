/**
 * @file ITCState.hpp
 * @brief Implementation of the Boussinesq thermal convection in a sphere
 * (Toroidal/Poloidal formulation)
 */

#ifndef QUICC_MODEL_BOUSSINESQ_SPHERE_TC_ITCSTATE_HPP
#define QUICC_MODEL_BOUSSINESQ_SPHERE_TC_ITCSTATE_HPP

// System includes
//
#include <string>

// Project includes
//
#include "QuICC/Generator/StateGenerator.hpp"
#include "QuICC/Model/IStateGeneratorBuilder.hpp"

namespace QuICC {

namespace Model {

namespace Boussinesq {

namespace Sphere {

namespace TC {

/**
 * @brief Implementation of the Boussinesq thermal convection sphere model
 * (Toroidal/Poloidal formulation)
 */
class ITCState : public IStateGeneratorBuilder<StateGenerator>
{
public:
   /**
    * @brief Constructor
    */
   ITCState() = default;

   /**
    * @brief Destructor
    */
   virtual ~ITCState() = default;

   /// Formulation used for vector fields
   virtual VectorFormulation::Id SchemeFormulation() override;

   /**
    * @brief Version string
    */
   std::string version() const final;

   /**
    * @brief Add the initial state generation equations
    *
    * @param spGen   Shared generator object
    */
   virtual void addStates(SharedStateGenerator spGen) override;

protected:
private:
};

} // namespace TC
} // namespace Sphere
} // namespace Boussinesq
} // namespace Model
} // namespace QuICC

#endif // QUICC_MODEL_BOUSSINESQ_SPHERE_TC_ITCSTATE_HPP
