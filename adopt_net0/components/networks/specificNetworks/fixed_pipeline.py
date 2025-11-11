import pandas as pd
import pyomo.environ as pyo
import pyomo.gdp as gdp
import copy
from warnings import warn

from ..genericNetworks.fluid import Fluid
from ...utilities import get_attribute_from_dict


class FixedPipeline(Fluid):

    def __init__(self, netw_data: dict):
        """
        Constructor

        :param dict netw_data: network data
        """
        super().__init__(netw_data)

    def fit_network_performance(self):
        """
        Fits network performance for fluid network (bounds and coefficients).
        """
        super().fit_network_performance()

    def _define_size_arc(self, b_arc, b_netw, node_from: str, node_to: str):
        """
        Defines the size of an arc

        :param b_arc: pyomo arc block
        :param b_netw: pyomo network block
        :param str node_from: node from which arc comes
        :param str node_to: node to which arc goes
        :return: pyomo arc block
        """

        coeff_ti = self.processed_coeff.time_independent

        if self.size_is_int:
            size_domain = pyo.NonNegativeIntegers
        else:
            size_domain = pyo.NonNegativeReals

        b_arc.para_size_max = pyo.Param(
            domain=size_domain,
            initialize=coeff_ti["size_max_arcs"].at[node_from, node_to],
        )

        b_arc.distance = self.distance.at[node_from, node_to]

        if self.existing and self.decommission == "impossible":
            # Decommissioning is not possible, size fixed
            b_arc.var_size = pyo.Param(
                within=size_domain,
                initialize=b_netw.para_size_initial[node_from, node_to],
            )
        else:
            # Bounds numerici espliciti per evitare GDP_Error
            size_min = float(pyo.value(b_netw.para_size_min))
            size_max = float(pyo.value(b_arc.para_size_max))

            # Variabile continua con bounds chiari
            b_arc.var_size = pyo.Var(bounds=(0, size_max), within=pyo.NonNegativeReals)

            # Variabile binaria che attiva la dimensione dell'arco
            b_arc.y_size = pyo.Var(within=pyo.Binary)

            # Vincoli per applicare logica "o 0 oppure in [min, max]"
            def size_lb_rule(b):
                return b.var_size >= size_min * b.y_size

            def size_ub_rule(b):
                return b.var_size <= size_max * b.y_size

            b_arc.size_lb_con = pyo.Constraint(rule=size_lb_rule)
            b_arc.size_ub_con = pyo.Constraint(rule=size_ub_rule)

        return b_arc

    def _define_capex_constraints_arc(self, b_arc, b_netw, node_from, node_to):
        """
        Defines the capex of an arc and corresponding constraints

        :param b_arc: pyomo arc block
        :param b_netw: pyomo network block
        :param str node_from: node from which arc comes
        :param str node_to: node to which arc goes
        :return: pyomo arc block
        """
        coeff_ti = self.processed_coeff.time_independent
        rated_capacity = coeff_ti["rated_capacity"]

        def init_capex(const):
            return (
                b_arc.var_capex_aux
                == (
                    b_arc.para_capex_gamma1
                    + b_arc.para_capex_gamma2 * b_arc.var_size
                    + b_arc.para_capex_gamma3 * b_arc.distance
                    + b_arc.para_capex_gamma4 * b_arc.var_size * b_arc.distance
                )
                * b_arc.y_size
            )

        # CAPEX aux:
        if self.existing and self.decommission == "impossible":
            if b_arc.var_size.value == 0:
                b_arc.const_capex_aux = pyo.Constraint(expr=b_arc.var_capex_aux == 0)
            else:
                b_arc.const_capex_aux = pyo.Constraint(rule=init_capex)
        elif (b_arc.para_capex_gamma1.value == 0) and (
            b_arc.para_capex_gamma3.value == 0
        ):
            b_arc.const_capex_aux = pyo.Constraint(rule=init_capex)
        else:
            b_arc.big_m_transformation_required = 1
            s_indicators = range(0, 2)

            def init_installation(dis, ind):
                if ind == 0:  # network not installed
                    dis.const_capex_aux = pyo.Constraint(expr=b_arc.var_capex_aux == 0)
                    dis.const_not_installed = pyo.Constraint(expr=b_arc.var_size == 0)
                else:  # network installed
                    dis.const_capex_aux = pyo.Constraint(rule=init_capex)

            b_arc.dis_installation = gdp.Disjunct(s_indicators, rule=init_installation)

            def bind_disjunctions(dis):
                return [b_arc.dis_installation[i] for i in s_indicators]

            b_arc.disjunction_installation = gdp.Disjunction(rule=bind_disjunctions)

        # CAPEX and CAPEX aux
        if self.existing:
            if not self.decommission == "impossible":
                b_arc.const_capex = pyo.Constraint(
                    expr=b_arc.var_capex
                    == (b_netw.para_size_initial[node_from, node_to] - b_arc.var_size)
                    * b_arc.para_decommissioning_cost_annual
                )
        else:
            b_arc.const_capex = pyo.Constraint(
                expr=b_arc.var_capex == b_arc.var_capex_aux
            )

        return b_arc
