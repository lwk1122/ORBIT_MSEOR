import math

from orbit_or.template_solvers import solve_with_template


def test_template_solver_handles_training_asset_count():
    result = solve_with_template(
        "Annual production of fighter jets is a_1=10 and a_2=15. "
        "Each training jet can train 5 pilots per year. How many trained "
        "pilots are available by the end of year 2?"
    )

    assert result.matched is True
    assert result.template_id == "direct_training_asset_count"
    assert result.status == "optimal"
    assert result.objective_value == 125


def test_template_solver_handles_rectangular_assignment_table():
    result = solve_with_template(
        """
        Find a job assignment plan that minimizes total working hours.

        | Worker | A | B | C | D |
        |--------|---|---|---|---|
        | I      | 9 | 4 | 3 | 7 |
        | II     | 4 | 6 | 5 | 6 |
        | III    | 5 | 4 | 7 | 5 |
        | IV     | 7 | 5 | 2 | 3 |
        | V      | 10| 6 | 7 | 4 |
        """
    )

    assert result.matched is True
    assert result.template_id == "assignment_min_cost"
    assert result.status == "optimal"
    assert result.objective_value == 14


def test_template_solver_handles_preference_assignment_goal_programming():
    result = solve_with_template(
        """
        A company needs to recruit three types of professionals to two regional
        branches. The company's personnel arrangement considers priorities:
        p_1: all three types of professionals needed are fully met; p_2:
        8000 recruited personnel meet their preferred specialty; p_3: 8000
        recruited personnel meet their preferred city. At least how many
        people cannot meet P3 under this priority?

        | Branch Location | Specialty | Demand |
        |-----------------|-----------|--------|
        | Donghai City    | 1         | 1000   |
        | Donghai City    | 2         | 2000   |
        | Donghai City    | 3         | 1500   |
        | Nanjiang City   | 1         | 2000   |
        | Nanjiang City   | 2         | 1000   |
        | Nanjiang City   | 3         | 1000   |

        | Type | Number of People | Suitable Specialty | Preferred Specialty | Preferred City |
        |------|------------------|--------------------|---------------------|----------------|
        | 1    | 1500             | 1,2                | 1                   | Donghai        |
        | 2    | 1500             | 2,3                | 2                   | Donghai        |
        | 3    | 1500             | 1,3                | 1                   | Nanjiang       |
        | 4    | 1500             | 1,3                | 3                   | Nanjiang       |
        | 5    | 1500             | 2,3                | 3                   | Donghai        |
        | 6    | 1500             | 3                  | 3                   | Nanjiang       |
        """
    )

    assert result.matched is True
    assert result.template_id == "preference_assignment_goal_programming"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 2500)


def test_template_solver_handles_max_flow_network():
    result = solve_with_template(
        """
        A telecommunications network spans 6 key points, from Point 0 to Point
        5. The objective is to find the maximum flow from Point 0 to Point 5
        without exceeding line capacities.

        - From Point 0 (Data Center): Can send data to Point 1 (14 GB/s),
          Point 2 (9 GB/s), Point 3 (8 GB/s), Point 4 (7 GB/s), and Point 5
          (8 GB/s).
        - From Point 1: Can send data to Point 0 (4 GB/s), Point 2 (9 GB/s),
          Point 3 (10 GB/s), and Point 5 (12 GB/s).
        - From Point 2: Can send data to Point 0 (4 GB/s), Point 1 (12 GB/s),
          Point 3 (13 GB/s), Point 4 (20 GB/s), and Point 5 (16 GB/s).
        - From Point 3: Can send data to Point 0 (10 GB/s), Point 1 (8 GB/s),
          Point 2 (12 GB/s), and Point 5 (18 GB/s).
        - From Point 4: Can send data to Point 0 (3 GB/s), Point 1 (13 GB/s),
          Point 2 (11 GB/s), Point 3 (20 GB/s), and Point 5 (2 GB/s).
        - From Point 5 (User Hub): Can send data back to Point 0 (17 GB/s),
          Point 1 (4 GB/s), Point 2 (8 GB/s), Point 3 (2 GB/s), and Point 4
          (12 GB/s).
        """
    )

    assert result.matched is True
    assert result.template_id == "max_flow_network"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 46)


def test_template_solver_handles_narrative_max_flow_network():
    result = solve_with_template(
        """
        There are four nodes connected by directed edges. Node 1, which could
        represent the starting point, is connected to two other nodes: There
        is an edge from node 1 to node 2 with a capacity of 8. There is an
        edge from node 1 to node 3 with a capacity of 7. Node 2 is an
        intermediate node: An edge leading from node 2 to node 3 with a
        capacity of 2. An edge leading from node 2 to node 4 with a capacity
        of 4. Node 3 has an edge from node 3 to node 4 with a substantial
        capacity of 12. Node 4 could represent the target or terminal. Find
        the corresponding maximum flow of the graph.
        """
    )

    assert result.matched is True
    assert result.template_id == "max_flow_network"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 13)


def test_template_solver_handles_city_max_flow_rows_without_from_prefix():
    result = solve_with_template(
        """
        A supply network must maximize flow from City 0 to City 3. Route
        capacities are:

        - From City 0 (Source): Can send supplies to City 1 (5 tons),
          City 2 (10 tons), and City 3 (1 ton).
        - City 1: Can receive and then send supplies to City 2 (1 ton)
          and City 3 (5 tons).
        - City 2: Can manage supplies from City 0 (2 tons), and then
          distribute to City 3 (10 tons).
        - City 3 (Destination): Can receive from City 0 (99 tons),
          City 1 (99 tons), and City 2 (99 tons).

        What is the maximum amount of supplies from the source to the
        destination without exceeding route capacities?
        """
    )

    assert result.matched is True
    assert result.template_id == "max_flow_network"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 16)


def test_template_solver_normalizes_max_flow_thousand_units():
    result = solve_with_template(
        """
        A traffic network reports capacities for vehicles in thousands. Compute
        the maximum amount of traffic in thousands from City 0 to City 3.

        - From City 0 (Source): Vehicles can flow to City 1 (18,000),
          and City 2 (5,000).
        - City 1: Vehicles can flow to City 3 (18,000).
        - City 2: Vehicles can flow to City 3 (5,000).
        - City 3 (Destination): Vehicles can return to City 0 (7,000).

        What is the maximum amount of traffic in thousands of vehicles per hour?
        """
    )

    assert result.matched is True
    assert result.template_id == "max_flow_network"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 23)


def test_template_solver_handles_named_source_with_station_number():
    result = solve_with_template(
        """
        An electrical grid must maximize the flow from the Power Plant to the
        Main Distribution Hub. Cable capacities are:

        - From the Power Plant (Station 0): Can transmit electricity to
          Station 1 (4 MW), Station 2 (12 MW), Station 3 (19 MW), Station 4
          (13 MW), and Station 5 (11 MW).
        - From Station 1: Can transmit to Station 2 (13 MW), Station 3
          (9 MW), and Station 4 (10 MW).
        - From Station 2: Can transmit to Station 1 (8 MW), Station 3 (5 MW),
          Station 4 (13 MW), and Station 5 (11 MW).
        - From Station 3: Can transmit to Station 1 (15 MW), Station 2
          (18 MW), Station 4 (6 MW), and Station 5 (7 MW).
        - From Station 4: Can transmit to Station 1 (17 MW), Station 2
          (20 MW), Station 3 (14 MW), and Station 5 (9 MW).
        - To the Main Distribution Hub (Station 5): Has incoming summary
          lines from upstream stations.

        What is the maximum flow from the source to the destination?
        """
    )

    assert result.matched is True
    assert result.template_id == "max_flow_network"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 38)


def test_template_solver_rejects_incomplete_max_flow_summary():
    result = solve_with_template(
        """
        The goal is to maximize the flow from Data Center 0 to Data Center 8.
        Capacities are described in a summary:

        - From Data Center 0 (Source): Data can be transmitted to Data Center 1
          (6 Tbps), Data Center 2 (20 Tbps), and Data Center 8 (20 Tbps).
        - From Data Center 1: Data can be transmitted to Data Centers ranging
          from 0 to 8 with various capacities up to 18 Tbps.
        - From Data Center 8 (Destination): It can receive data from all other
          centers.

        What is the maximum flow from the source to the destination?
        """
    )

    assert result.matched is False


def test_template_solver_handles_latex_permutation_flow_shop():
    result = solve_with_template(
        r"""
        A fabric dyeing plant has 3 dyeing vats. Each batch of fabric must be
        dyed in sequence in each vat: first, second, and third vats. The time
        required in hours to dye batch i in vat j is given in the matrix:

        $$
        \left(\begin{array}{ccc}
        3 & 1 & 1 \\
        2 & 1.5 & 1 \\
        3 & 1.2 & 1.3 \\
        2 & 2 & 2 \\
        2.1 & 2 & 3
        \end{array}\right)
        $$

        Schedule the dyeing operations to minimize the completion time of the
        last batch.
        """
    )

    assert result.matched is True
    assert result.template_id == "permutation_flow_shop_scheduling"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 14.1)


def test_template_solver_handles_table_permutation_flow_shop():
    result = solve_with_template(
        """
        Three products must first be processed on machine 1, then sequentially
        on machines 2 and 3. The order of processing the products on each
        machine should remain the same. Minimize the total processing cycle.

        | Product | Machine 1 | Machine 2 | Machine 3 |
        |---------|-----------|-----------|-----------|
        | Product 1 | 2 | 3 | 1 |
        | Product 2 | 4 | 2 | 3 |
        | Product 3 | 3 | 5 | 2 |
        """
    )

    assert result.matched is True
    assert result.template_id == "permutation_flow_shop_scheduling"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 14)


def test_template_solver_handles_security_maximin_net_revenue():
    result = solve_with_template(
        """
        Before a championship, there are 5 types of securities available for
        sale. The price of each share is fixed and the payoff is contingent on
        the outcome. The Share Limit is the maximum number of shares one can
        purchase. There are five countries, Argentina, Brazil, England, Spain,
        Germany. Security 1 has a price of 0.75 and a share limit of 10. The
        payoff for this security is $1 in Argentina, $1 in Brazil, $1 in
        England, 0 in Germany, and 0 in Spain. Security 2 is priced at 0.35
        with a share limit of 5. The payoff is 0 for Argentina, Brazil, or
        England, but it's $1 for Germany and Spain. Security 3 has a price of
        0.40 and a share limit of 10. The payoff is $1 for Argentina, $1 for
        Spain, and $1 for England, and 0 payoff for Germany and Brazil.
        Security 4, with a price of 0.75 and a share limit of 10, has payoff
        of $1 across all countries except for Spain. Security 5 is priced at
        0.65, has a share limit of 5, and has payoff of $1 in Brazil, Germany
        and Spain, 0 in Argentina and England. Find the maximum worst-case
        revenue.
        """
    )

    assert result.matched is True
    assert result.template_id == "security_maximin_net_revenue_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 1.0)
    artifact = result.artifact or {}
    assert artifact["budget_constraint_present"] is False
    assert artifact["diagnostics"][0]["issue_type"] == "missing_budget_or_normalization"


def test_template_solver_handles_security_budget_constraint():
    result = solve_with_template(
        """
        There are two countries, Alpha, Beta. Security 1 has a price of 0.4
        and a share limit of 10. The payoff is $1 in Alpha and 0 in Beta.
        Security 2 has a price of 0.4 and a share limit of 10. The payoff is
        0 in Alpha and $1 in Beta. The total budget is $4. Find the maximum
        worst-case revenue.
        """
    )

    assert result.matched is True
    assert result.template_id == "security_maximin_net_revenue_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 1.0)
    artifact = result.artifact or {}
    assert artifact["budget_constraint_present"] is True
    assert math.isclose(artifact["budget_limit"], 4.0)
    assert math.isclose(artifact["total_purchase_cost"], 4.0)
    assert artifact["diagnostics"] == []


def test_template_solver_handles_box_uncertain_resource_lp():
    result = solve_with_template(
        """
        A plant produces products A and B. Unit profits are 5 and 4,
        respectively. Each unit uses nominal resource coefficients of 2 and 3,
        respectively. The coefficients are uncertain under box uncertainty
        with deviations 0.5 and 0.5, respectively. The resource capacity is
        100. Demand limits for A and B are 40 and 40, respectively. Find the
        robust production plan that maximizes profit.
        """
    )

    assert result.matched is True
    assert result.template_id == "robust_resource_capacity_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 200.0)
    assert math.isclose((result.variable_values or {})["produce_A"], 40.0)
    artifact = result.artifact or {}
    assert artifact["uncertainty_type"] == "box"
    assert artifact["modeling_experience"]["family"] == "robust_optimization"


def test_template_solver_handles_budget_uncertain_resource_lp():
    result = solve_with_template(
        """
        A plant produces products A and B. Unit profits are 5 and 4,
        respectively. Each unit uses nominal resource coefficients of 2 and 3,
        respectively. The coefficients are uncertain under budget uncertainty
        with deviations 1 and 1, respectively. Gamma is 1. The resource
        capacity is 100. Demand limits for A and B are 40 and 40,
        respectively. Find the robust production plan that maximizes profit.
        """
    )

    assert result.matched is True
    assert result.template_id == "robust_resource_capacity_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 166.66666666666666)
    assert math.isclose((result.variable_values or {})["produce_A"], 33.333333333333336)
    artifact = result.artifact or {}
    assert artifact["uncertainty_type"] == "budget"
    assert "Bertsimas-Sim" in artifact["robust_counterpart"]


def test_template_solver_handles_tsp_routing():
    result = solve_with_template(
        """
        A sales representative must visit four distinct cities and return to
        the starting city. They must visit each city exactly once and minimize
        the minimum total travel cost.

        - From City A, the cost is 19 units to City B, 45 units to City C,
          and 30 units to City D.
        - Traveling from City B, it costs 19 units to reach City A,
          89 units to City C, and 46 units to City D.
        - From City C, the journey costs 45 units to City A, 89 units to
          City B, and merely 11 units to City D.
        - Lastly, from City D, the travel expenses are 30 units to City A,
          46 units to City B, and 11 units to City C.
        """
    )

    assert result.matched is True
    assert result.template_id == "tsp_routing_enum"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 121)


def test_template_solver_reports_tsp_subtour_relaxation_gap():
    result = solve_with_template(
        """
        A salesperson must visit each city exactly once and return to the
        starting city while minimizing total travel cost.

        - From City A, the cost is 1 units to City B, 100 units to City C,
          and 100 units to City D.
        - From City B, the cost is 1 units to City A, 100 units to City C,
          and 100 units to City D.
        - From City C, the cost is 100 units to City A, 100 units to City B,
          and 1 units to City D.
        - From City D, the cost is 100 units to City A, 100 units to City B,
          and 1 units to City C.
        """
    )

    assert result.matched is True
    assert result.template_id == "tsp_routing_enum"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 202)
    artifact = result.artifact or {}
    assert artifact["requires_subtour_elimination"] is True
    assert math.isclose(artifact["assignment_relaxation_objective"], 4)
    assert sorted(sorted(cycle) for cycle in artifact["assignment_relaxation_cycles"]) == [
        ["A", "B"],
        ["C", "D"],
    ]


def test_template_solver_handles_symmetric_tsp_upper_triangular_matrix():
    result = solve_with_template(
        """
        A traveling salesman must visit 7 customers at 7 different locations,
        with the symmetric distance matrix as follows:

        |  | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
        | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
        | 1 | - | 86 | 49 | 57 | 31 | 69 | 50 |
        | 2 |  | - | 68 | 79 | 93 | 24 | 5 |
        | 3 |  |  | - | 16 | 7 | 72 | 67 |
        | 4 |  |  |  | - | 90 | 69 | 1 |
        | 5 |  |  |  |  | - | 86 | 59 |
        | 6 |  |  |  |  |  | - | 81 |

        Formulate a mathematical program to determine the visiting order
        starting and ending at location 1 to minimize the travel distance.
        """
    )

    assert result.matched is True
    assert result.template_id == "tsp_routing_enum"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 153)


def test_template_solver_handles_set_cover_table():
    result = solve_with_template(
        """
        A convenience supermarket needs to open the minimum number of chain
        stores so every residential area is within range.

        | Area Code | Residential Areas within 800 m Radius |
        |-----------|----------------------------------------|
        | A         | A, C, E, G, H, I                       |
        | B         | B, H, I                                |
        | C         | A, C, G, H, I                          |
        | D         | D, J                                   |
        | E         | A, E, G                                |
        | F         | F, J, K                                |
        | G         | A, C, E, G                             |
        | H         | A, B, C, H, I                          |
        | I         | A, B, C, H, I                          |
        | J         | D, F, J, K, L                          |
        | K         | F, J, K, L                             |
        | L         | J, K, L                                |
        """
    )

    assert result.matched is True
    assert result.template_id == "set_cover_enum"
    assert result.status == "optimal"
    assert result.objective_value == 3


def test_template_solver_handles_minimum_vertex_cover():
    result = solve_with_template(
        """
        A graph has 10 vertices labeled with lowercase letter from 'a' to 'j'.
        The vertex cover problem asks for the smallest set of vertices such
        that every edge is incident to at least one selected vertex. Vertex
        'a' connects to vertices 'f', 'e', and 'b'. Vertex 'b' connect to
        vertices 'a', 'g', and 'c'. Vertex 'c' connects to vertices 'b', 'h',
        and 'd'. Vertex 'd' connects to vertices 'c', 'i', and 'e'. Vertex
        'e' connects to vertices 'd', 'j', and 'a'. The vertices inside the
        pentagon ('f', 'g', 'h', 'i', 'j') are all interconnected. Find the
        minimum vertex cover.
        """
    )

    assert result.matched is True
    assert result.template_id == "minimum_vertex_cover_enum"
    assert result.status == "optimal"
    assert result.objective_value == 7


def test_template_solver_diagnoses_invalid_vertex_cover_candidate():
    result = solve_with_template(
        """
        A graph has vertices labeled with lowercase letter from 'a' to 'j'.
        The red-colored vertices ('a', 'c', 'd', 'f', 'g', and 'j') suggest a
        possible vertex cover. The vertex cover problem asks for the smallest
        set of vertices such that every edge is incident to at least one
        selected vertex. Vertex 'a' connects to vertices 'f', 'e', and 'b'.
        Vertex 'b' connect to vertices 'a', 'g', and 'c'. Vertex 'c' connects
        to vertices 'b', 'h', and 'd'. Vertex 'd' connects to vertices 'c',
        'i', and 'e'. Vertex 'e' connects to vertices 'd', 'j', and 'a'. The
        vertices inside the pentagon ('f', 'g', 'h', 'i', 'j') are all
        interconnected. Find the minimum vertex cover.
        """
    )

    assert result.matched is True
    assert result.template_id == "minimum_vertex_cover_enum"
    assert result.status == "optimal"
    assert result.objective_value == 7
    artifact = result.artifact or {}
    assert artifact["max_proven_infeasible_cover_size"] == 6
    candidate = artifact["candidate_vertex_sets"][0]
    assert candidate["label"] == "red_vertices"
    assert candidate["is_vertex_cover"] is False
    assert ["h", "i"] in candidate["uncovered_edges"]
    assert artifact["diagnostics"][0]["issue_type"] == "invalid_candidate_vertex_cover"


def test_template_solver_handles_continuous_width_cutting_stock():
    result = solve_with_template(
        """
        A paper mill receives three orders for rolls of paper.

        | Order Number | Width (meters) | Length (meters) |
        | :---: | :---: | :---: |
        | 1 | 0.5 | 1000 |
        | 2 | 0.7 | 3000 |
        | 3 | 0.9 | 2000 |

        The mill produces rolls of paper with standard widths of 1 meter and
        2 meters. Assuming the length of the rolls is unlimited and can be
        spliced to reach the required length, how should the rolls be cut to
        minimize the area of waste?
        """
    )

    assert result.matched is True
    assert result.template_id == "continuous_width_cutting_stock_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 600)


def test_template_solver_handles_integer_length_cutting_stock():
    result = solve_with_template(
        """
        A steel reinforcement workshop produces a batch of steel bars,
        consisting of 90 pieces of 3 meters in length and 60 pieces of 4
        meters in length. Each piece of raw steel bar used is 10 meters in
        length. How can the raw material be cut most efficiently? Establish a
        linear programming model to minimize the total waste.
        """
    )

    assert result.matched is True
    assert result.template_id == "integer_length_cutting_stock_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 20)


def test_template_solver_handles_interval_contract_covering():
    result = solve_with_template(
        """
        A factory must rent warehouse space for the next 4 months. The
        required warehouse area is:

        | Month | 1 | 2 | 3 | 4 |
        |-------|------|------|------|------|
        | Required area | 1500 | 1000 | 2000 | 1200 |

        The factory can sign rental contracts of different lengths.

        | Contract length (months) | 1 | 2 | 3 | 4 |
        |--------------------------|----|----|----|----|
        | Rental fee per 100 sqm | 4000 | 7500 | 10500 | 13000 |

        At least two different contracts must be signed. If a 4-month contract
        is chosen, then no 1-month contract may be chosen. The number of
        different warehouse contracts signed cannot exceed 3. Demand for each
        month must be satisfied without shortage or excess. Minimize total
        rental cost.
        """
    )

    assert result.matched is True
    assert result.template_id == "interval_contract_covering_milp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 208000)
    assert result.artifact and result.artifact["used_lengths"] == [1, 2, 3]


def test_template_solver_handles_supplier_order_lot_mix():
    result = solve_with_template(
        """
        A restaurant needs to order dining tables from three different
        suppliers, A, B, and C. The cost of ordering each dining table from
        Supplier A is 120, from Supplier B is 110, and from Supplier C is 100.
        The restaurant needs to minimize the total cost of the order.

        Each order from Supplier A will include 20 tables, while each order
        from Suppliers B and C will include 15 tables. The number of orders
        must be an integer. The restaurant needs to order at least 150 tables
        and no more than 600 tables.

        If the restaurant decides to order tables from Supplier A, it must also
        order at least 30 tables from Supplier B. If the restaurant decides to
        order tables from Supplier B, it must also order tables from Supplier C.
        """
    )

    assert result.matched is True
    assert result.template_id == "procurement_lot_mix_milp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 15000)
    assert result.variable_values == {"count_C": 10.0}


def test_template_solver_handles_warehouse_truck_material_mix():
    result = solve_with_template(
        """
        A production base needs to extract raw materials from warehouses A and
        B every day. The required raw materials are: at least 240 pieces of raw
        material A, at least 80 kg of raw material B, and at least 120 tons of
        raw material C. Each truck from warehouse A can transport back 4 pieces
        of raw material A, 2 kg of raw material B, 6 tons of raw material C,
        with a freight cost of 200 yuan per truck; each truck from warehouse B
        can transport back 7 pieces of raw material A, 2 kg of raw material B,
        2 tons of raw material C, with a freight cost of 160 yuan per truck.
        How many trucks should be dispatched daily to minimize freight cost?
        """
    )

    assert result.matched is True
    assert result.template_id == "procurement_lot_mix_milp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 6800)
    assert result.variable_values == {"count_A": 10.0, "count_B": 30.0}


def test_template_solver_handles_integer_transport_mode_mix():
    result = solve_with_template(
        """
        A company can choose from the following three methods: motorcycle,
        small truck, and large truck. Each motorcycle trip produces 40 units
        of pollution, each small truck trip produces 70 units of pollution,
        and each large truck trip produces 100 units of pollution. The goal is
        to minimize total pollution. The company can only choose two out of
        these three transportation methods. The number of motorcycle trips
        cannot exceed 8. Each motorcycle trip can transport 10 units, each
        small truck trip can transport 20 units, and each large truck trip can
        transport 50 units. The company needs to transport at least 300 units.
        The total number of trips must be less than or equal to 20.
        """
    )

    assert result.matched is True
    assert result.template_id == "integer_resource_mix_milp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 600)
    assert result.variable_values == {"count_large truck": 6.0}


def test_template_solver_handles_integer_truck_capacity_mix():
    result = solve_with_template(
        """
        A transportation company has two types of trucks, Type A and Type B.
        Type A trucks have 20 cubic meters of refrigerated capacity and 40
        cubic meters of non-refrigerated capacity. Type B trucks have the same
        total capacity, but the capacities for refrigerated and
        non-refrigerated cargo are equal. A grocer needs to rent trucks to
        transport 3000 cubic meters of refrigerated cargo and 4000 cubic
        meters of non-refrigerated cargo. The rental cost per kilometer for
        Type A trucks is 30, while the rental cost per kilometer for Type B
        trucks is 40. Minimize the total cost.
        """
    )

    assert result.matched is True
    assert result.template_id == "integer_resource_mix_milp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 4170)
    assert result.variable_values == {"count_Type A": 51.0, "count_Type B": 66.0}


def test_template_solver_handles_two_vehicle_share_mix_minimizing_one_mode():
    result = solve_with_template(
        """
        A village hosts a banquet and provides bike and car transportation for
        everyone. A bike can take 3 people while a car can take 5 people. Since
        cars are more expensive, at most 40% of the vehicles can be cars. If
        the village needs to transport at least 500 people, how many of each
        vehicle should be used to minimize the total number of bikes needed?
        """
    )

    assert result.matched is True
    assert result.template_id == "integer_resource_mix_milp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 80)
    assert result.variable_values == {"count_bike": 80.0, "count_car": 52.0}


def test_template_solver_handles_two_trip_modes_with_share_and_upper_bound():
    result = solve_with_template(
        """
        Ducks need to be taken to shore to be cleaned either by boat or by
        canoe. A boat can take 10 ducks per trip while a canoe can take 8 ducks
        per trip. The boats take 20 minutes per trip while the canoes take 40
        minutes per trip. There can be at most 12 boat trips and at least 60%
        of the trips should be by canoe. If at least 300 ducks need to be taken
        to shore, how many of each transportation method should be used to
        minimize the total amount of time needed to transport the ducks?
        """
    )

    assert result.matched is True
    assert result.template_id == "integer_resource_mix_milp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 1160)
    assert result.variable_values == {"count_boat": 12.0, "count_canoe": 23.0}


def test_template_solver_handles_vehicle_count_with_pollution_cap():
    result = solve_with_template(
        """
        A tourism company can buy sedans or buses to add to their fleet of
        vehicles. A sedan can seat 50 tourists per day but results in 10 units
        of pollution. A bus can seat 250 tourists per day but results in 40
        units of pollution. The city has limited this company to producing at
        most 800 units of pollutants per day. The company must take care of at
        least 4600 customers per day. How many sedans and buses should the
        company purchase to decrease the total number of vehicles needed?
        """
    )

    assert result.matched is True
    assert result.template_id == "integer_resource_mix_milp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 19)
    assert result.variable_values == {"count_bus": 19.0}


def test_template_solver_handles_worker_shift_count_mix():
    result = solve_with_template(
        """
        An accounting firm employs part time workers and full time workers. Full
        time workers work 8 hours per shift while part time workers work 4 hours
        per shift. In addition, full time workers are paid $300 per shift while
        part time workers are paid $100 per shift. Currently, the accounting
        firm has a project requiring 500 hours of labor. If the firm has a
        budget of $15000, how many of each type of worker should be scheduled to
        minimize the total number of workers.
        """
    )

    assert result.matched is True
    assert result.template_id == "worker_shift_count_mix_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 100)
    assert result.variable_values == {
        "full_time_workers": 25.0,
        "part_time_workers": 75.0,
    }


def test_template_solver_handles_xy_integer_salary_minimization():
    result = solve_with_template(
        r"""
        A Human Resources manager is planning the allocation of new hires
        between two departments, $X$ and $Y$. The company needs to hire at least
        10 new employees. However, due to space limitations in the office, the
        combined effort for these departments, calculated as 3 times the number
        of hires for department X plus 4 times the number of hires for
        department Y, must not exceed 40. Given that the annual salary per
        employee for department $X$ is $\$50000$, and for department $Y$ it's
        $\$60000$, and the HR manager aims to minimize total salaries while
        meeting all constraints(X,Y are integers).
        """
    )

    assert result.matched is True
    assert result.template_id == "xy_two_variable_integer_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 500000)
    assert result.variable_values == {"X": 10.0}


def test_template_solver_handles_xy_integer_ratio_offset_minimization():
    result = solve_with_template(
        """
        A marketing firm is planning the distribution of resources between two
        campaigns: X and Y. The total resources allocated to both campaigns
        cannot exceed 1000 units due to budget constraints. The resources
        allocated to campaign X must be at least twice as many as those
        allocated to campaign Y plus an additional 300 units. The costs
        associated with each unit of resource for campaigns X and Y are 4 and 3
        units respectively. The allocations for both campaigns must be whole
        numbers, and the goal is to find the minimum total cost.
        """
    )

    assert result.matched is True
    assert result.template_id == "xy_two_variable_integer_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 1200)
    assert result.variable_values == {"X": 300.0}


def test_template_solver_handles_xy_integer_resource_and_yield_minimization():
    result = solve_with_template(
        """
        A farmer is planning to plant two crops, $X$ and $Y$. The planting must
        be done in whole numbers. Each unit of crop $X$ requires 5 units of
        water and each unit of crop $Y$ requires 3 units of water. The total
        amount of water available is limited to 100 units. The combined yield
        from twice the units of crop $X$ and once the unit of crop $Y$ should be
        at least 30. The cost associated with each unit for crops X and Y are 4
        and 2 respectively, and the farmer aims to minimize this total cost.
        """
    )

    assert result.matched is True
    assert result.template_id == "xy_two_variable_integer_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 60)
    assert result.variable_values == {"X": 15.0}


def test_template_solver_handles_xyz_integer_hiring_minimization():
    result = solve_with_template(
        r"""
        A human resources manager is planning to hire new employees for three
        different departments: $X$, $Y$, and $Z$. The cost of hiring an employee
        for department $X$ is $\$5000$, for department $Y$ it's $\$4000$, and
        for department $Z$ it's $\$6000$. The company has a policy to hire
        exactly 50 new employees this year. The total experience score,
        calculated as 5 times the number of hires in department X plus 3 times
        the number of hires in department Y plus 7 times the number of hires in
        department Z, must be at least 200. Moreover, there must be at least 10
        employees hired for department X and no more than 20 employees can be
        hired for department Y. All hires are whole numbers, and the goal is the
        minimum total cost.
        """
    )

    assert result.matched is True
    assert result.template_id == "xyz_three_variable_integer_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 230000)
    assert result.variable_values == {"X": 30.0, "Y": 20.0}


def test_template_solver_handles_table_capacity_space_mix():
    result = solve_with_template(
        """
        In a science fair, there are two types of tables that can be used to
        display experiments. At the circular tables, 4 poster boards and 5
        participants can fit around the table to cater to 8 guests. At the
        rectangular tables, 4 poster boards and 4 participants can fit around
        the table to cater to 12 guests. Each circular table takes up 15 units
        of space while each rectangular table takes up 20 units of space. The
        fair must be able to fit at least 500 participants and 300 poster
        boards. If the fair has available 1900 units of space, how many of each
        type of table should be set up to maximize catered guests?
        """
    )

    assert result.matched is True
    assert result.template_id == "table_capacity_space_mix_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 1080)
    assert result.variable_values == {
        "count_circular_tables": 60.0,
        "count_rectangular_tables": 50.0,
    }


def test_template_solver_handles_fixed_charge_machine_assignment():
    result = solve_with_template(
        r"""
        There are 10 different parts, and they can all be processed on machine
        \( A \), machine \( B \), or machine \( C \). As long as any part is
        processed on a machine, a one-time setup cost is incurred with
        \( d_A = 100 \), \( d_B = 135 \), and \( d_C = 200 \). One piece of
        each part must be processed. If the 1st part is processed on machine
        \( A \), then the 2nd part must be processed on machine \( B \) or
        \( C \); conversely, if the 1st part is processed on machine \( B \) or
        \( C \), then the 2nd part must be processed on machine \( A \). Parts
        3, 4, and 5 must be processed on machines A, B, and C respectively. The
        number of parts processed on machine \( C \) should not exceed 3 types.
        Minimize total cost.

        | Machine/Part | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
        |--------------|---|---|---|---|---|---|---|---|---|----|
        | A | $10$ | $20$ | $30$ | $40$ | $50$ | $60$ | $70$ | $80$ | $90$ | $100$ |
        | B | $15$ | $25$ | $35$ | $45$ | $55$ | $65$ | $75$ | $85$ | $95$ | $105$ |
        | C | $20$ | $30$ | $40$ | $50$ | $60$ | $70$ | $80$ | $90$ | $100$ | $110$ |
        """
    )

    assert result.matched is True
    assert result.template_id == "fixed_charge_machine_assignment_milp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 1005)
    assert result.variable_values["assign_part_3_machine_A"] == 1.0
    assert result.variable_values["assign_part_4_machine_B"] == 1.0
    assert result.variable_values["assign_part_5_machine_C"] == 1.0


def test_template_solver_handles_cart_share_min_count():
    result = solve_with_template(
        """
        A resort can move guests using either golf carts or pull carts. A golf
        cart can take 4 guests while a pull cart can take 1 guest. At most 60%
        of carts can be golf carts. If at least 80 guests need to be moved,
        how many of each cart should be used to minimize the total number of
        carts?
        """
    )

    assert result.matched is True
    assert result.template_id == "integer_resource_mix_milp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 29)


def test_template_solver_handles_container_capacity_max_mix():
    result = solve_with_template(
        """
        A farm can transport grapes using small crates or large crates. A small
        crate can take 200 grapes while a large crate can take 500 grapes. At
        least 3 times as many small crates must be used than large crates. At
        most 100 small crates and at most 50 large crates are available. At
        most 60 crates total can be loaded, and at least 10 large crates must
        be used. How many crates should be used to maximize total capacity?
        """
    )

    assert result.matched is True
    assert result.template_id == "container_capacity_max_mix_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 16500)


def test_template_solver_handles_staff_shift_count_wording():
    result = solve_with_template(
        """
        A mail room has full-time staff and part-time staff. Full-time staff
        works 40 hours per week and gets paid $1280. Part-time staff works 15
        hours per week and gets paid $450. The mail room needs 1000 hours of
        labor and has a budget of $31500. How many staff members are needed to
        decrease the total number of staff?
        """
    )

    assert result.matched is True
    assert result.template_id == "worker_shift_count_mix_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 40)


def test_template_solver_handles_two_supplement_integer_diet():
    result = solve_with_template(
        """
        Calcium and Magnesium are found in two health supplements, health
        supplement A and health supplement B. One serving of health supplement
        A contains 30 grams of Calcium and 50 grams of Magnesium. One serving
        of health supplement B contains 60 grams of Calcium and 10 grams of
        Magnesium. The cost per health supplement for health supplement A is
        $14 and the cost per health supplement for health supplement B is $25.
        A patient must consume these two health supplements every day to get at
        least 400 grams of Calcium and 50 grams of Magnesium. Determine how
        much servings of each supplement the patient needs to minimize her
        daily cost.
        """
    )

    assert result.matched is True
    assert result.template_id == "two_supplement_integer_diet_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 175)


def test_template_solver_handles_minimum_lower_bound_allocation():
    result = solve_with_template(
        """
        A company allocates resources to product X, product Y, and product Z.
        Product X generates $5 per unit, product Y generates $3 per unit, and
        product Z generates $2 per unit. Product X needs at least 200 units,
        product Y needs a minimum of 150 units, and product Z needs a minimum
        of 100 units. Allocations must be whole numbers. Find the minimum
        total cost.
        """
    )

    assert result.matched is True
    assert result.template_id == "minimum_lower_bound_allocation_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 1650)


def test_template_solver_handles_two_item_weighted_score_min_cost():
    result = solve_with_template(
        """
        A manufacturer must choose between tanks and aircraft for shipment.
        Each tank weighing 20 tons and each aircraft weighing 30 tons can be
        loaded, but total weight is limited to 50000 tons. The effectiveness
        score is calculated as twice the number of tanks plus one time the
        number of aircraft and must yield at least 2000. The cost per unit for
        a tank is $5000 and for an aircraft is $3000. Quantities must be
        integers. What is the minimum cost?
        """
    )

    assert result.matched is True
    assert result.template_id == "two_item_weighted_score_min_cost_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 5000000)


def test_template_solver_handles_xy_quantity_difference_bounds():
    result = solve_with_template(
        """
        A company makes products X and Y. The cost associated with each unit of
        resource for products X and Y are 5 and 2 units respectively. The total
        quantity of X and Y cannot exceed 1000. Quantity of X minus twice
        quantity of Y should be at least 300, and three times quantity of
        product X along with that of product Y should be at least 500. X can
        range from 0-600 units, while Y can range from 0-500 units. The
        quantities must be whole numbers, and the goal is to find the minimum
        total cost.
        """
    )

    assert result.matched is True
    assert result.template_id == "xy_two_variable_integer_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 1500)


def test_template_solver_handles_advertising_media_mix():
    result = solve_with_template(
        """
        A food company would like to run its commercials on three streaming
        platforms: Pi TV, Beta Video and Gamma Live. On Pi TV, a commercial
        costs $1200 and attracts 2000 viewers. On Beta Video, a commercial
        costs $2000 and attracts 5000 viewers. On Gamma Live, a commercial
        costs $4000 and attracts 9000 viewers. Beta Video limits the number of
        commercials from a single company to 8. At most a third of all
        commercials should occur on Gamma Live and a minimum of 20% should
        occur on Pi TV. If the weekly budget is $20000, how many commercials
        should be run in each choice to maximize audience?
        """
    )

    assert result.matched is True
    assert result.template_id == "advertising_media_mix_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 46000)


def test_template_solver_handles_two_product_machine_inventory_surplus():
    result = solve_with_template(
        """
        A company operates two machines to produce two fertilizers: liquid and
        solid. Planning may use fractional lots when appropriate. Producing
        one lot of liquid requires 50 minutes on Machine 1 and 30 minutes on
        Machine 2. Producing one lot of solid requires 24 minutes on Machine 1
        and 33 minutes on Machine 2. On-hand inventories are 30 lots of liquid
        and 90 lots of solid. This week, Machine 1 has 40 available hours and
        Machine 2 has 35 available hours. Forecast demand for this week is 75
        lots of liquid and 95 lots of solid. The company aims to maximize the
        total ending inventory of liquid and solid at the end of the week.
        """
    )

    assert result.matched is True
    assert result.template_id == "two_product_machine_inventory_surplus_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 1.25)


def test_template_solver_handles_three_project_integer_linear_min_cost():
    result = solve_with_template(
        """
        A telecommunications company is planning to allocate resources among
        three projects: Project X1, Project X2, and Project X3. The objective
        is to minimize total cost where costs are associated as $10 per unit
        for project X1, $20 per unit for project X2 and $30 per unit for
        project X3. The combined resource allocation for project X1 and twice
        that allocated to project X2 should not exceed 2000 units. The sum of
        thrice the allocation for project X1 and quadruple that of project X3
        should be at least 1000 units. The difference between the resources
        allocated to project X2 and five times those allocated to project X3
        should not surpass 500 units. Allocations X1, X2, and X3 must be in
        whole numbers. What is the minimum total cost?
        """
    )

    assert result.matched is True
    assert result.template_id == "three_project_integer_linear_min_cost"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 3340)


def test_template_solver_handles_small_symbolic_training_hours():
    result = solve_with_template(
        """
        A sports coach is planning weekly training hours among four groups:
        x1, x2, x3, and x4. The objective is to minimize fatigue calculated as
        10 for group x1, 15 for group x2, 12 for group x3 and 20 for group x4.
        The combined training hours of group x1 and group x2 should not exceed
        8 hours. The combined hours of group x3 and group x4 cannot be more
        than 10 hours. The sum of x1 and x3 should be at least 5 hours.
        Similarly, the combined time spent on x2 and x4 must be at least 6
        hours. Each session requires whole numbers of training hours. What is
        the minimum possible fatigue score?
        """
    )

    assert result.matched is True
    assert result.template_id == "small_symbolic_integer_min_cost"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 146)


def test_template_solver_handles_symbolic_project_difference_constraints():
    result = solve_with_template(
        """
        A telecommunications company is planning to allocate resources to four
        projects: x1, x2, x3, and x4. The cost associated with these projects
        are $50 per unit for project x1, $60 per unit for project x2, $30 per
        unit for project x3 and $40 per unit for project x4. The combined
        resource allocation for x1 and x2 cannot exceed 5000 units. The
        combined resource allocation for x3 and x4 cannot exceed 2000 units.
        The difference between the allocations of resources in project x1 and
        project x2 should be at least 1000 units. The difference between the
        allocations of resources in project x3 and project x4 should not
        exceed 500 units. x1: [0,3000], x2: [0,2500], x3: [0,1500],
        x4: [0,1200]. The allocations must be whole numbers. Minimize cost.
        """
    )

    assert result.matched is True
    assert result.template_id == "small_symbolic_integer_min_cost"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 50000)


def test_template_solver_handles_symbolic_machinery_group_constraints():
    result = solve_with_template(
        """
        A construction company has four types of machinery: X, Y, Z, and W.
        The rental cost per day is $4000 for X, $3000 for Y, $6000 for Z, and
        $5000 for W. The company can only rent a total of 50 machines from
        group X and Y combined, and a total of 40 machines from group Z and W
        combined. At least 30 machines must be rented from group X or Z
        combined, and at least 20 machines must be rented from group Y or W
        combined. There are maximums of 30 for X, 25 for Y, 20 for Z, and 15
        for W. All rentals are integers and the company minimizes rental cost.
        """
    )

    assert result.matched is True
    assert result.template_id == "small_symbolic_integer_min_cost"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 180000)


def test_template_solver_handles_three_project_weighted_symbolic_min_cost():
    result = solve_with_template(
        """
        A company allocates resources to three projects: X, Y, and Z. Costs are
        $10 per unit for project X, $20 per unit for project Y, and $30 per
        unit for project Z. The combined resource allocation for all three
        projects cannot exceed 10000 units. The sum of twice the allocation for
        project X plus thrice the allocation for project Y and the allocation
        for project Z must be at least 5000 units. The difference between
        allocations of projects X and Y plus the allocation of project Z should
        not exceed 3000 units. Allocations must be whole numbers. Minimize
        total cost.
        """
    )

    assert result.matched is True
    assert result.template_id == "small_symbolic_integer_min_cost"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 25000)


def test_template_solver_handles_two_test_probe_mix():
    result = solve_with_template(
        """
        A chemical company uses two tests, a salinity test and a pH test. Each
        unit of the salinity test requires three probes. Each unit of the pH
        test requires two probes. The company must perform at least 250 pH
        tests. In total, at least 400 tests must be performed. There must be
        at most 1.5 times more pH tests than salinity tests. Minimize the total
        number of probes used.
        """
    )

    assert result.matched is True
    assert result.template_id == "two_test_probe_mix_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 1001)


def test_template_solver_handles_infeasible_furnace_purchase_mix():
    result = solve_with_template(
        """
        A high rise building is buying new model and old model furnaces. A new
        model furnace can heat 10 apartments and consumes 200 kWh per day. An
        old model can heat 15 apartments and consumes 250 kWh per day. At most
        35% of the furnaces can be the old model and at least 5 new model
        furnaces should be used. The building needs to heat at least 200
        apartments and has 3500 kWh of electricity available. Minimize the
        total number of furnaces.
        """
    )

    assert result.matched is True
    assert result.template_id == "furnace_purchase_min_count_ilp"
    assert result.status == "infeasible"


def test_template_solver_handles_two_ingredient_mix_profit():
    result = solve_with_template(
        """
        A shop sells cat paw snacks and gold shark snacks in bulk. It prepares
        two snack mix products. The first mix contains 20% cat paw snacks and
        80% gold shark snacks. The second mix contains 35% cat paw snacks and
        65% gold shark snacks. The store has on hand 20 kg of cat paw snacks
        and 50 kg of gold shark snacks. The profit per kg of the first mix is
        $12 and the profit per kg of the second mix is $15. Maximize profit.
        """
    )

    assert result.matched is True
    assert result.template_id == "two_ingredient_mix_profit_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 960)


def test_template_solver_handles_two_team_capacity_max_mix():
    result = solve_with_template(
        """
        A lawn mowing service uses small teams and large teams. A small team
        requires 3 employees and can mow 50 sq ft of lawn. A large team
        requires 5 employees and can mow 80 sq ft of lawn. The company has 150
        employees available. The number of small teams must be at least 3 times
        as much as the number of large teams. There has to be at least 6 large
        teams and at least 10 small teams. Maximize the amount of lawn mowed.
        """
    )

    assert result.matched is True
    assert result.template_id == "two_team_capacity_max_mix_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 2480)


def test_template_solver_handles_two_food_ratio_protein_max():
    result = solve_with_template(
        """
        A woman needs to eat two meal preps, a smoothie and a protein bar. Each
        smoothie contains 2 units of protein and 300 calories. Each protein bar
        contains 7 units of protein and 250 calories. The woman must eat 2
        times more protein bars than smoothies. She can consume at most 2000
        calories. Maximize her protein intake.
        """
    )

    assert result.matched is True
    assert result.template_id == "two_food_ratio_protein_max_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 32)


def test_template_solver_handles_bombing_success_probability_dp():
    result = solve_with_template(
        """
        Certain strategic bomber groups are tasked with destroying enemy
        military targets. The target has four key parts, and destroying at
        least two of them will suffice. Bomb stockpile: A maximum of 28 heavy
        bombs and 12 light bombs can be used. Fuel limit: Total fuel
        consumption must not exceed 10,000 liters. When carrying heavy bombs,
        each liter of fuel allows a distance of 2 km, whereas with light bombs,
        each liter allows 3 km. Each aircraft can only carry one bomb per trip,
        and each bombing run requires fuel for the round trip: each liter of
        fuel allows 4 km when the aircraft is empty, plus 100 liters for both
        takeoff and landing per trip. Maximize the probability of success.

        | Key Part | Distance from Airport (km) | Probability of Destruction per Heavy Bomb | Probability of Destruction per Light Bomb |
        |----------|----------------------------|------------------------------------------|------------------------------------------|
        | 1        | 450                        | 0.03                                     | 0.08                                     |
        | 2        | 480                        | 0.10                                     | 0.11                                     |
        | 3        | 540                        | 0.05                                     | 0.12                                     |
        | 4        | 600                        | 0.05                                     | 0.09                                     |
        """
    )

    assert result.matched is True
    assert result.template_id == "bombing_success_probability_dp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 0.576593, abs_tol=1e-6)


def test_template_solver_handles_two_option_resource_max_mix():
    result = solve_with_template(
        """
        A dog school trains labradors and golden retrievers to deliver
        newspapers. A labrador can deliver 7 newspapers at a time and requires
        5 small bone treats for service. A golden retriever can deliver 10
        newspapers at a time and requires 6 small bone treats per service. The
        school only has 1500 small bone treats available. At least 50 golden
        retrievers must be used and at most 60% of the dogs can be labradors.
        Maximize the number of newspapers delivered.
        """
    )

    assert result.matched is True
    assert result.template_id == "two_option_resource_max_mix_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 2500)


def test_template_solver_handles_two_option_keyboard_infeasible():
    result = solve_with_template(
        """
        A keyboard manufacturer makes mechanical and standard keyboards.
        Mechanical keyboards cost five units of plastic and two units of
        solder whereas a standard keyboard costs two units of plastic and one
        unit of solder. The manufacturer aims to have five times as many
        mechanical than standard keyboards. There must be at least 30 standard
        keyboards. The company has available 1000 units of plastic and 250
        units of solder. Maximize the total number of keyboards.
        """
    )

    assert result.matched is True
    assert result.template_id == "two_option_resource_max_mix_ilp"
    assert result.status == "infeasible"


def test_template_solver_handles_group_lower_bound_fatigue_allocation():
    result = solve_with_template(
        """
        A coach allocates weekly training hours among three groups: X, Y, and
        Z. The total number of training hours is constrained to a maximum of
        1000 hours. Group X requires at least 200 hours, group Y needs at
        least 150 hours, and group Z requires no less than 250 hours. The
        fatigue scores are 2 for group X, 3 for group Y, and 4 for group Z
        respectively. Hours are whole numbers and the coach minimizes fatigue.
        """
    )

    assert result.matched is True
    assert result.template_id == "minimum_lower_bound_allocation_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 1850)


def test_template_solver_handles_portfolio_fee_lower_bounds():
    result = solve_with_template(
        r"""
        A finance manager is allocating a total investment fund of \$100,000
        across four portfolios: $X$, $Y$, $Z$ and $W$. For portfolio $X$, the
        rate is 0.05, for portfolio $Y$ it's 0.07, for portfolio $Z$ it's
        0.06 and for portfolio $W$ it's 0.09. Portfolio X requires at least
        \$20,000, Portfolio Y requires at least \$15,000, Portfolio Z requires
        at least \$25,000, and Portfolio W requires at least \$40,000. Minimize
        the total annual management fee.
        """
    )

    assert result.matched is True
    assert result.template_id == "portfolio_fee_lower_bounds_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 7150)


def test_template_solver_handles_real_estate_weighted_roi_min_cost():
    result = solve_with_template(
        r"""
        A real estate investor is considering residential ($x$), commercial
        ($y$), and industrial ($z$) properties. The cost per unit of investment
        in these property types are \$300, \$200, and \$500 respectively. The
        total number of units invested cannot exceed 50. ROI is calculated as
        5 times the number of residential units plus 3 times the number of
        commercial units plus 10 times the number of industrial units. The
        combined ROI must be at least 100. The investor wants at least as many
        residential units as commercial ones. Investments are whole numbers and
        the goal is minimum total cost.
        """
    )

    assert result.matched is True
    assert result.template_id == "real_estate_weighted_roi_min_cost_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 5000)


def test_template_solver_handles_four_unit_pair_strength_min_cost():
    result = solve_with_template(
        r"""
        A military general allocates resources between unit types X1, X2, X3
        and X4. The total number of units for type X1 and X2 combined cannot
        exceed 200, and the total number for X3 and X4 combined cannot exceed
        150. Combined strength calculated as 5 times the unit count for type X1
        plus 10 times the unit count for type X2 must be at least 1000.
        Another strength calculated as 7.5 times the unit count for type X3
        plus 5 times the unit count for type X4 must be at least 750. Each
        unit of types X1, X2, X3 and X4 requires \$5000, \$7000, \$10000 and
        \$3000 respectively. X1, X2, X3, X4 are integers. Minimize total cost.
        """
    )

    assert result.matched is True
    assert result.template_id == "four_unit_pair_strength_min_cost_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 1150000)


def test_template_solver_handles_healthcare_department_allocation():
    result = solve_with_template(
        """
        In a healthcare management scenario, a hospital allocates resources
        across four departments: X1, X2, X3, and X4. The costs being 0.5, 0.2,
        0.3, and 0.4 units for X1, X2, X3 and X4 respectively. The combined
        resource allocation for departments X1 and X2 cannot exceed 100 units.
        The sum of allocations for departments X2 and X3 must be at least 50
        units. The difference between the allocations for department X3 and
        department X4 must not exceed 30 units. Considering certain strategic
        objectives quantified by the equation: (10 times allocation for
        department X1) + (15 times allocation for department X2) - (5 times
        allocation for department X3) - (20 times allocation for department X4)
        must be at least 150 units. Department X1 can't receive more than 100
        units; Department X2 can't receive more than 80 units; Department X3
        can't receive more than 60 units; Department X4 can't receive more
        than 50 units. Allocations are whole numbers. Minimize total cost.
        """
    )

    assert result.matched is True
    assert result.template_id == "healthcare_department_allocation_min_cost_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 10)


def test_template_solver_handles_grain_inventory_arbitrage():
    result = solve_with_template(
        """
        A trading company sells grain. It has a warehouse with a capacity of
        5000 dan. On January 1, the company has 1000 dan of grain in stock and
        20,000 yuan in funds. The purchased grains will be delivered in the
        same month but can only be sold in the next month. The company hopes to
        have an inventory of 2000 dan at the end of the quarter and wants to
        maximize total profit.

        | Month | Purchase Price (yuan/dan) | Selling Price (yuan/dan) |
        |-------|---------------------------|--------------------------|
        | 1     | 2.85                      | 3.10                     |
        | 2     | 3.05                      | 3.25                     |
        | 3     | 2.90                      | 2.95                     |
        """
    )

    assert result.matched is True
    assert result.template_id == "grain_inventory_arbitrage_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, -700)


def test_template_solver_handles_fixed_activation_quota_product_plan():
    result = solve_with_template(
        """
        A company plans to produce products A1, A2, A3 for 22 days. Determine a
        production plan that maximizes total revenue while accommodating fixed
        activation costs and minimum production batch constraints.

        | Product | A1 | A2 | A3 |
        | :---: | :---: | :---: | :---: |
        | Maximum Demand | 5300 | 4500 | 5400 |
        | Selling Price | $124$ | $109$ | $115$ |
        | Production Cost | $73.30$ | $52.90$ | $65.40$ |
        | Production Quota | 500 | 450 | 550 |

        | Product | A1 | A2 | A3 |
        | :---: | :---: | :---: | :---: |
        | Activation Cost | $170000$ | $150000$ | $100000$ |

        $$
        \\begin{array}{c|ccc}
        Product & A_{1} & A_{2} & A_{3} \\\\
        \\hline
        Minimum Batch & 20 & 20 & 16
        \\end{array}
        $$
        """
    )

    assert result.matched is True
    assert result.template_id == "fixed_activation_quota_product_plan_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 270290)


def test_template_solver_handles_multi_machine_process_profit():
    result = solve_with_template(
        """
        Product I can be processed on any equipment for A and B; Product II can
        be processed on any A equipment but only on B1; Product III can only be
        processed on A2 and B2. Arrange the optimal production plan to maximize
        profit with effective machine hours and operating costs at full
        capacity.

        | Equipment  | Product I | Product II | Product III | Effective Machine Hours | Operating Costs at Full Capacity (Yuan) |
        |------------|-----------|------------|-------------|--------------------------|------------------------------------------|
        | A1         | 5         | 10         |             | 6000                     | 300                                      |
        | A2         | 7         | 9          | 12          | 10000                    | 321                                      |
        | B1         | 6         | 8          |             | 4000                     | 250                                      |
        | B2         | 4         |            | 11          | 7000                     | 783                                      |
        | B3         | 7         |            |             | 4000                     | 200                                      |
        | Raw Material Cost (Yuan/Unit) | 0.25 | 0.35       | 0.50       |                          |                                          |
        | Unit Price (Yuan/Unit)        | 1.25 | 2.00       | 2.80       |                          |                                          |
        """
    )

    assert result.matched is True
    assert result.template_id == "multi_machine_process_profit_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 1146.4142, rel_tol=0.05)


def test_template_solver_handles_farm_operating_plan():
    result = solve_with_template(
        """
        A certain farm has 100 hectares of land and 15,000 yuan in funds for
        production development. The labor force situation on the farm is 3,500
        person-days in autumn and winter, and 4,000 person-days in spring and
        summer. If labor is not fully utilized, they can work externally,
        earning 2.1 yuan/person-day in spring and summer and 1.8 yuan/person-day
        in autumn and winter. The farm cultivates soybeans, corn, and wheat,
        and raises dairy cows and chickens. Raising dairy cows involves an
        investment of 400 yuan per cow, uses 1.5 hectares per cow, 100
        autumn/winter person-days, 50 spring/summer person-days, and earns 400
        yuan per cow. Chickens require 3 yuan per chicken, 0.6 autumn/winter
        person-days, 0.3 spring/summer person-days, earn 2 yuan per chicken,
        and capacity is 3000 chickens. The cow barn can accommodate up to 32
        dairy cows. Determine the farm operating plan to maximize annual net
        income.

        | Item | Soybean | Corn | Wheat |
        |------|---------|------|-------|
        | Person-days (Autumn/Winter) | 20 | 35 | 10 |
        | Person-days (Spring/Summer) | 50 | 75 | 40 |
        | Annual Net Income (Yuan/hectare) | 175 | 300 | 120 |
        """
    )

    assert result.matched is True
    assert result.template_id == "farm_operating_plan_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 20241.8, rel_tol=0.05)


def test_template_solver_handles_narrative_transportation_distribution():
    result = solve_with_template(
        """
        There are two coal yards A and B, each receiving no less than 80 tons
        and 100 tons of coal per month, respectively. They are responsible for
        supplying coal to three residential areas, which need 55 tons, 75 tons,
        and 50 tons of coal per month, respectively. Coal yard A is located 10
        kilometers, 5 kilometers, and 6 kilometers from these three residential
        areas. Coal yard B is located 4 kilometers, 8 kilometers, and 15
        kilometers from these three residential areas. How should these coal
        yards distribute coal to minimize the ton-kilometers of transportation?
        """
    )

    assert result.matched is True
    assert result.template_id == "narrative_transportation_distribution_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 1030)
    assert result.variable_values == {
        "ship_A_to_destination_2": 30.0,
        "ship_A_to_destination_3": 50.0,
        "ship_B_to_destination_1": 55.0,
        "ship_B_to_destination_2": 45.0,
    }


def test_template_solver_handles_fixed_charge_transshipment():
    result = solve_with_template(
        """
        There are m=2 production points with output a_1 = 100 and a_2 = 150.
        The material is shipped to n=2 demand points with b_1 = 80 and
        b_2 = 120. Shipments must pass through one of p=2 intermediate
        marshaling stations. If station k is used, a fixed cost f_k is
        incurred, where f_1 = 10 and f_2 = 15. Station capacities are
        q_1 = 100 and q_2 = 100. The unit transportation costs from production
        points to stations are c_{11}=2, c_{12}=3, c_{21}=4, c_{22}=1.
        The unit transportation costs from stations to demand points are
        c'_{11}=3, c'_{12}=2, c'_{21}=1, and c'_{22}=4. Determine the
        transportation plan that minimizes total cost.
        """
    )

    assert result.matched is True
    assert result.template_id == "fixed_charge_transshipment_milp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 685)
    assert result.variable_values == {
        "ship_source_1_to_station_1": 100.0,
        "ship_source_2_to_station_2": 100.0,
        "ship_station_1_to_destination_2": 100.0,
        "ship_station_2_to_destination_1": 80.0,
        "ship_station_2_to_destination_2": 20.0,
        "open_station_1": 1.0,
        "open_station_2": 1.0,
    }


def test_template_solver_handles_transportation_table_with_truck_capacity():
    result = solve_with_template(
        """
        An Italian transportation company needs to move empty containers from
        warehouses to major ports. The container inventory is:

        |  | Empty Containers |
        |:---:|:---:|
        | Verona | 10 |
        | Perugia | 12 |
        | Rome | 20 |
        | Pescara | 24 |
        | Taranto | 18 |
        | Lamezia | 40 |

        The demand at the ports is:

        |  | Container Demand |
        |:---:|:---:|
        | Genoa | 20 |
        | Venice | 15 |
        | Ancona | 25 |
        | Naples | 33 |
        | Bari | 21 |

        The cost is proportional to distance, with a rate of 30 euros per
        kilometer. Each truck can carry up to 2 containers. The distances are:

        |  | Genoa | Venice | Ancona | Naples | Bari |
        |:---:|:---:|:---:|:---:|:---:|:---:|
        | Verona | 290 km | 115 km | 355 km | 715 km | 810 km |
        | Perugia | 380 km | 340 km | 165 km | 380 km | 610 km |
        | Rome | 505 km | 530 km | 285 km | 220 km | 450 km |
        | Pescara | 655 km | 450 km | 155 km | 240 km | 315 km |
        | Taranto | 1010 km | 840 km | 550 km | 305 km | 95 km |
        | Lamezia | 1072 km | 1097 km | 747 km | 372 km | 333 km |
        """
    )

    assert result.matched is True
    assert result.template_id == "transportation_table_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 468510)


def test_template_solver_handles_fixed_charge_facility_location():
    result = solve_with_template(
        """
        EfficientDistro needs to decide which distribution centers to open and
        how to ship goods to retail stores while minimizing opening and
        transportation costs.

        Opening Costs for Each Distribution Center (in dollars):
        - Center 1: $151,000
        - Center 2: $192,000
        - Center 3: $114,000
        - Center 4: $171,000

        Transportation Cost Per Unit from Each Distribution Center to Retail Stores (in dollars):
        - From Center 1: $5 to Store 1, $5 to Store 2, $2 to Store 3, $3 to Store 4
        - From Center 2: $3 to Store 1, $3 to Store 2, $5 to Store 3, $4 to Store 4
        - From Center 3: $3 to Store 1, $5 to Store 2, $2 to Store 3, $4 to Store 4
        - From Center 4: $2 to Store 1, $4 to Store 2, $5 to Store 3, $1 to Store 4

        Demand of Each Retail Store (in units):
        - Store 1: 859
        - Store 2: 713
        - Store 3: 421
        - Store 4: 652

        Supply Capacity of Each Distribution Center (in units):
        - Center 1: 1,547
        - Center 2: 1,656
        - Center 3: 1,274
        - Center 4: 1,882
        """
    )

    assert result.matched is True
    assert result.template_id == "fixed_charge_facility_location"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 273940)


def test_template_solver_handles_fixed_charge_substitution_production():
    result = solve_with_template(
        """
        Red Star Plastics Factory produces six distinct types of plastic
        containers. Each container type has a volume, market demand, and unit
        variable production cost.

        | Container Type (Code) | 1 | 2 | 3 | 4 | 5 | 6 |
        | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
        | Volume ($\\text{cm}^3$) | 1500 | 2500 | 4000 | 6000 | 9000 | 12000 |
        | Market Demand (units) | 500 | 550 | 700 | 900 | 400 | 300 |
        | Unit Variable Production Cost (Yuan/unit) | 5 | 8 | 10 | 12 | 16 | 18 |

        Activating any container type incurs a fixed setup cost of 1200 Yuan.
        If production is insufficient, the factory may use larger or equal
        volume containers as substitutes. Minimize the total cost while fully
        meeting demand.
        """
    )

    assert result.matched is True
    assert result.template_id == "fixed_charge_substitution_production"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 43200)


def test_template_solver_handles_periodic_production_inventory_lp():
    result = solve_with_template(
        """
        A company must determine how many sailboats should be produced during
        each of the next four quarters. The demand during each of the next four
        quarters is as follows: first quarter, 40 sailboats; second quarter, 60
        sailboats; third quarter, 75 sailboats; fourth quarter, 25 sailboats.
        At the beginning of the first quarter, the company has an inventory of
        10 sailboats. During each quarter, it can produce up to 40 sailboats
        with regular-time labor at a total cost of $400 per sailboat. By having
        employees work overtime during a quarter, it can produce additional
        sailboats with overtime labor at a total cost of $450 per sailboat. At
        the end of each quarter, a carrying or holding cost of $20 per sailboat
        is incurred. Use linear programming to minimize production and inventory
        costs.
        """
    )

    assert result.matched is True
    assert result.template_id == "periodic_production_inventory_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 78450)


def test_template_solver_handles_multi_period_workforce_production_plan():
    result = solve_with_template(
        """
        A company needs to create an optimal production and human resources plan
        for a six-month period to maximize total net profit.

        **Initial Conditions:**
        - Initial Workforce: 1,000 employees
        - Initial Inventory: 15,000 units

        **Revenue and Cost Structure:**
        - **Sales Price:** 300 Yuan per unit sold.
        - **Raw Material Cost:** 90 Yuan per unit for units produced in-house.
        - **Outsourcing Cost:** 200 Yuan per unit for finished tables.
        - **Inventory Holding Cost:** 15 Yuan per unit held at the end of a month.
        - **Backorder Cost:** 35 Yuan per unit carried over to the next month.

        **Labor and Production Parameters:**
        - Each in-house unit requires 5 labor hours to produce.
        - Each worker provides 160 regular working hours per month.
        - The company pays a regular wage of 30 Yuan/hour.
        - Total overtime hours per month cannot exceed 20 hours per worker.
        - The overtime wage is 40 Yuan/hour.
        - The cost to hire a new worker is 5,000 Yuan.
        - The cost to fire a worker is 8,000 Yuan.

        **Terminal Condition:**
        - The ending inventory must be at least 10,000 units.
        - All backorders must be cleared, so ending backorders must be zero.

        **Forecasted Demand:**
        | Month | January | February | March | April | May | June |
        |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
        | Demand Forecast | 20,000 | 40,000 | 42,000 | 35,000 | 19,000 | 18,500 |
        """
    )

    assert result.matched is True
    assert result.template_id == "multi_period_workforce_production_plan_milp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 10349920)
    assert result.variable_values["workforce_January"] == 156.0
    assert result.variable_values["fired_January"] == 844.0
    assert result.variable_values["inventory_June"] == 10000.0


def test_template_solver_handles_tool_repair_replacement_milp():
    result = solve_with_template(
        """
        A factory needs to use a special tool over n planning stages. At stage
        j, r_j specialized tools are needed. At the end of each stage, used
        tools may be sent for repair before they can be reused. There are two
        repair methods: slow repair costs b per tool and takes p stages, while
        fast repair costs c per tool and takes q stages. If repaired tools
        cannot meet the need, new tools must be purchased at cost a.

        n = 10
        r = [3, 5, 2, 4, 6, 5, 4, 3, 2, 1]
        a = 10
        b = 1
        c = 3
        p = 3
        q = 1
        """
    )

    assert result.matched is True
    assert result.template_id == "tool_repair_replacement_milp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 168)


def test_template_solver_handles_reliability_spares_allocation():
    result = solve_with_template(
        r"""
        An electronic system is composed of 3 types of components. The
        system's operational reliability is the product of the reliabilities
        of each component. By installing spare parts, the reliability of each
        component can be improved. The total budget for all spare parts is
        limited to 150 yuan, and the weight limit is 20 kg.

        \begin{tabular}{|c|c|c|c|}
        \hline
        \textbf{Component Number} & \textbf{1} & \textbf{2} & \textbf{3} \\ \hline
        \textbf{Number of Spares} &             &             &             \\ \hline
        0                & 0.5         & 0.6         & 0.7         \\ \hline
        1                & 0.6         & 0.75        & 0.9         \\ \hline
        2                & 0.7         & 0.95        & 1.0         \\ \hline
        3                & 0.8         & 1.0         & 1.0         \\ \hline
        4                & 0.9         & 1.0         & 1.0         \\ \hline
        5                & 1.0         & 1.0         & 1.0         \\ \hline
        \textbf{Unit Price (yuan)}  & 20           & 30           & 40           \\ \hline
        \textbf{Unit Weight (kg)}  & 2            & 4            & 6            \\ \hline
        \end{tabular}
        """
    )

    assert result.matched is True
    assert result.template_id == "reliability_spares_allocation"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 0.6075)


def test_template_solver_handles_inventory_arbitrage_lp():
    result = solve_with_template(
        """
        A store plans to formulate the purchasing and sales plan for a product
        for the first quarter. The warehouse capacity of the store can store up
        to 500 units of the product, and there are 200 units in stock at the
        end of this year. The store purchases goods once at the beginning of
        each month. The purchasing and selling prices are:

        | Month | 1 | 2 | 3 |
        | :---: | :---: | :---: | :---: |
        | Purchasing Price (Yuan) | 8 | 6 | 9 |
        | Selling Price (Yuan) | 9 | 8 | 10 |

        Determine how many units should be purchased and sold each month to
        maximize the total profit.
        """
    )

    assert result.matched is True
    assert result.template_id == "inventory_arbitrage_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 4100)


def test_template_solver_handles_multi_product_inventory_backlog_ilp():
    result = solve_with_template(
        """
        The contract reservations for the next year for products I, II, and III
        of a certain factory in each quarter are shown below.

        | Product | 1    | 2    | 3    | 4    |
        |---------|------|------|------|------|
        | I       | 1500 | 1000 | 2000 | 1200 |
        | II      | 1500 | 1500 | 1200 | 1500 |
        | III     | 1000 | 2000 | 1500 | 2500 |

        At the beginning of the first quarter, there is no inventory for these
        three products, and it is required to have 150 units in stock for each
        product by the end of the fourth quarter. The factory has 15,000
        production hours per quarter, and each unit of products I, II, and III
        requires 2, 4, and 3 hours respectively. Product I cannot be produced
        in the second quarter. If products cannot be delivered on time, a
        compensation of 20 yuan per unit per quarter delay is required for
        products I and II, while for product III, the compensation is 10 yuan.
        Additionally, the inventory cost is 5 yuan per unit per quarter. How
        should the factory schedule production to minimize total cost?
        """
    )

    assert result.matched is True
    assert result.template_id == "multi_product_inventory_backlog_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 10755)


def test_template_solver_handles_workforce_training_delay_ilp():
    result = solve_with_template(
        """
        A factory produces two types of food, I and II, and currently has 50
        skilled workers. One skilled worker can produce $10 kg / h$ of food I
        or $6 kg / h$ of food II. The factory has decided to train 50 new
        workers by the end of the 8th week. A worker works $40 h$ per week, and
        a skilled worker can train up to three new workers in two weeks. During
        the training period, both the skilled worker and the trainees do not
        participate in production. The weekly wage of a skilled worker is 360
        yuan, the weekly wage of a trainee during the training period is 120
        yuan, and after training, the wage is 240 yuan per week. During the
        transition period, the factory can arrange some workers to work $60 h$
        per week, with a weekly wage of 540 yuan. If booked food cannot be
        delivered on time, the compensation fee for each week of delay per kg is
        0.5 yuan for food I and 0.6 yuan for food II. Minimize total cost.

        | Week | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
        |------|---|---|---|---|---|---|---|---|
        | I    | 10000 | 10000 | 12000 | 12000 | 16000 | 16000 | 20000 | 20000 |
        | II   | 6000 | 7200 | 8400 | 10800 | 10800 | 12000 | 12000 | 12000 |
        """
    )

    assert result.matched is True
    assert result.template_id == "workforce_training_delay_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 219888)


def test_template_solver_handles_continuous_nutrition_table_mix():
    result = solve_with_template(
        """
        Suppose an animal needs at least 700 g of protein, 30 g of minerals,
        and 100 mg of vitamins daily. There are five feeds available, and the
        nutritional content and price of each feed are shown below. Formulate a
        linear programming model that meets the animal's growth needs while
        minimizing the cost of selecting the feed.

        | Feed | Protein (g) | Minerals (g) | Vitamins (mg) | Price | Feed | Protein (g) | Minerals (g) | Vitamins (mg) | Price |
        |------|-------------|--------------|---------------|-------|------|-------------|--------------|---------------|-------|
        | 1    | 3           | 1            | 0.5           | 0.2   | 4    | 6           | 2            | 2             | 0.3   |
        | 2    | 2           | 0.5          | 1             | 0.7   | 5    | 18          | 0.5          | 0.8           | 0.8   |
        | 3    | 1           | 0.2          | 0.2           | 0.4   |      |             |              |               |       |
        """
    )

    assert result.matched is True
    assert result.template_id == "continuous_nutrition_mix_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 32.43589743589744)
    assert set(result.variable_values or {}) == {"amount_4", "amount_5"}


def test_template_solver_handles_integer_diet_lp():
    result = solve_with_template(
        """
        Keep the cost as low as possible while meeting nutrition targets.
        - Steak: It gives you 14 grams of protein, 23 grams of carbohydrates,
          and 63 calories for $4.
        - Tofu: It offers 2 grams of protein, 13 grams of carbohydrates,
          and 162 calories for $6.
        - Chicken: It packs 17 grams of protein, 13 grams of carbohydrates,
          and 260 calories for $6.
        - Broccoli: It provides 3 grams of protein, 1 gram of carbohydrates,
          and 55 calories for $8.
        - Rice: It gives 15 grams of protein, 23 grams of carbohydrates,
          and 231 calories for $8.
        - Spinach: It provides 2 grams of protein, 8 grams of carbohydrates,
          and 297 calories for $5.

        Ensure at least 83 grams of protein, 192 grams of carbohydrates,
        and 2089 calories. What is the minimum cost?
        """
    )

    assert result.matched is True
    assert result.template_id == "integer_diet_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 57)


def test_template_solver_handles_cost_first_integer_diet_lp():
    result = solve_with_template(
        """
        Minimize cost while meeting nutrition.
        - Food_1: Costs $4, provides 17 grams of protein,
          8 grams of carbohydrates, and 237 calories.
        - Food_2: Costs $2, provides 4 grams of protein,
          24 grams of carbohydrates, and 213 calories.
        - Food_3: Costs $6, provides 7 grams of protein,
          27 grams of carbohydrates, and 133 calories.
        - Food_4: Costs $2, provides 14 grams of protein,
          16 grams of carbohydrates, and 118 calories.
        - Food_5: Costs $6, provides 13 grams of protein,
          1 gram of carbohydrates, and 136 calories.
        - Food_6: Costs $8, provides 1 gram of protein,
          13 grams of carbohydrates, and 225 calories.

        Requirements are at least 76 grams of protein, 173 grams of
        carbohydrates, and 1751 calories.
        """
    )

    assert result.matched is True
    assert result.template_id == "integer_diet_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 22)


def test_template_solver_handles_cheapest_cost_diet_lp():
    result = solve_with_template(
        """
        Determine the most cost-effective combination. What is the cheapest
        cost to achieve the dietary goals?
        - Food_1: Delivers 3 grams of protein, 16 grams of carbohydrates,
          and 96 calories for $9.
        - Food_2: Provides 17 grams of protein, 24 grams of carbohydrates,
          and 182 calories for $9.
        - Food_3: Offers 16 grams of protein, 27 grams of carbohydrates,
          and 114 calories for just $2.
        - Food_4: Contains 8 grams of protein, 16 grams of carbohydrates,
          and 208 calories for $9.
        - Food_5: Supplies 6 grams of protein, 6 grams of carbohydrates,
          and 236 calories for $5.

        Consume at least 100 grams of protein, 180 grams of carbohydrates,
        and 1796 calories.
        """
    )

    assert result.matched is True
    assert result.template_id == "integer_diet_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 32)


def test_template_solver_handles_narrative_diet_with_carbs_abbreviation():
    result = solve_with_template(
        """
        Build the least expensive meal plan from these foods.

        - Chicken: It gives you 18 grams of protein, 5 grams of carbohydrates,
          and 202 calories for just $1.
        - Rice: With 14 grams of protein, 13 grams of carbs, and 234 calories,
          Rice is a bit pricier at $7.
        - Eggs: Eggs offer 18 grams of protein, along with 4 grams of carbs
          and 220 calories for $4.
        - Broccoli: A great source of protein at 7 grams, with 15 grams of
          carbs and 247 calories, this vegetable will cost you $3.
        - Lentils: They provide 15 grams of protein, 17 grams of carbs, and
          88 calories for only $1.
        - Apples: They offer 8 grams of protein, 13 grams of carbohydrates,
          and 77 calories for $2.

        The meal plan needs at least 70 grams of protein, 187 grams of
        carbohydrates, and 2181 calories. What is the minimum total cost?
        """
    )

    assert result.matched is True
    assert result.template_id == "integer_diet_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 16)


def test_template_solver_handles_product_mix_capacity_table():
    result = solve_with_template(
        """
        A factory produces two types of microcomputers, A and B, and wants to
        maximize total profit.

        | Process | Model |  | Maximum Weekly Processing Capacity |
        | :---: | :---: | :---: | :---: |
        |  | A | B |  |
        | I (hours / unit) | 4 | 6 | 150 |
        | II (hours / unit) | 3 | 2 | 70 |
        | Profit ($ per unit) | 300 | 450 |  |

        At least 10 units of Model A and at least 15 units of Model B must be
        produced per week.
        """
    )

    assert result.matched is True
    assert result.template_id == "product_mix_table_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 11250)


def test_template_solver_handles_product_mix_goal_wording_with_empty_headers():
    result = solve_with_template(
        """
        A factory produces two models of microcomputers, A and B. Each model
        requires the same two processes. Given the factory's business goals:
        p_1: The total weekly profit should not be less than 10,000 yuan;
        p_2: at least 10 units of model A and at least 15 units of model B
        must be produced each week. Formulate the mathematical model.

        | Process | Model | | Maximum Weekly Processing Capacity |
        | :---: | :---: | :---: | :---: |
        | | $A$ | $B$ | |
        | I (hours/unit) | 4 | 6 | 150 |
        | II (hours/unit) | 3 | 2 | 70 |
        | Profit (yuan/unit) | 300 | 450 | |
        """
    )

    assert result.matched is True
    assert result.template_id == "product_mix_table_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 11250)
    assert set(result.variable_values or {}) == {"produce_A", "produce_B"}


def test_template_solver_handles_three_product_capacity_table():
    result = solve_with_template(
        """
        A factory plans to produce three products, I, II, and III. Each product
        must be processed on equipment A, B, and C. How can equipment capacity
        be fully utilized to maximize production profit?

        | Equipment Code | I  | II | III | Effective Monthly Equipment Hours |
        |----------------|----|----|-----|----------------------------------|
        | A              | 8  | 2  | 10  | 300                              |
        | B              | 10 | 5  | 8   | 400                              |
        | C              | 2  | 13 | 10  | 420                              |
        | Unit Product Profit (per thousand yuan) | 3 | 2 | 2.9 | |
        """
    )

    assert result.matched is True
    assert result.template_id == "product_mix_table_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 135.26666666666668)
    assert set(result.variable_values or {}) == {"produce_I", "produce_II", "produce_III"}


def test_template_solver_handles_narrative_product_mix_with_resource_limits():
    result = solve_with_template(
        """
        A toy company manufactures three types of tabletop golf toys. The
        high-end type requires 17 hours of manufacturing labor, 8 hours of
        inspection, and yields a profit of 300 yuan per unit. The mid-range
        type requires 10 hours of labor, 4 hours of inspection, and yields a
        profit of 200 yuan per unit. The low-end type requires 2 hours of
        labor, 2 hours of inspection, and yields a profit of 100 yuan per unit.
        Available labor hours are 1000, and available inspection hours are 500.
        Demand is no more than 50 units for the high-end type, no more than 80
        units for the mid-range type, and no more than 150 units for the
        low-end type. Determine the production plan to maximize profit.
        """
    )

    assert result.matched is True
    assert result.template_id == "narrative_product_mix_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 25000)
    assert result.variable_values == {"produce_mid-range": 80.0, "produce_low-end": 90.0}


def test_template_solver_handles_package_revenue_mix():
    result = solve_with_template(
        """
        A store wants to clear out 200 shirts and 100 pairs of pants from last
        season. They introduce two promotional packages. Package A includes one
        shirt and two pairs of pants, priced at 30. Package B includes three
        shirts and one pair of pants, priced at 50. The store does not want to
        sell fewer than 20 A packages and 10 B packages. How many of each
        package should be sold to maximize revenue?
        """
    )

    assert result.matched is True
    assert result.template_id == "narrative_product_mix_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 3600)
    assert result.variable_values == {"produce_A": 20.0, "produce_B": 60.0}


def test_template_solver_handles_table_resource_cost_product_mix():
    result = solve_with_template(
        """
        A company is producing two products (X and Y). Production is measured in
        standardized batches, and the weekly plan may involve fractional
        batches when appropriate.

        | Item | Machine Time (minutes) | Craftsman Time (minutes) |
        | :---: | :---: | :---: |
        | X | 13 | 20 |
        | Y | 19 | 29 |

        The company has 40 hours of machine time available in the next working
        week, but only 35 hours of craftsman time. The cost of machine time is
        10 per hour, and the cost of craftsman time is 2 per hour. For each
        batch produced, the revenue for product X is 20, and the revenue for
        product Y is 30. A contract requires at least 10 batches of product X.
        Formulate a linear programming model to maximize profit.
        """
    )

    assert result.matched is True
    assert result.template_id == "narrative_product_mix_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 1866.3793103448277)
    assert math.isclose((result.variable_values or {})["produce_X"], 10)


def test_template_solver_handles_ratio_and_storage_product_mix():
    result = solve_with_template(
        """
        A company produces liquid product A and liquid product B. Each kilogram
        of product A sold generates a profit of 30, while each kilogram of
        product B sold generates a profit of 10. The company can allocate a
        maximum of 40 hours per week for production. Producing one kilogram of
        product A requires 6 hours, while producing one kilogram of product B
        requires 3 hours. Market demand requires that the output of product B
        must be at least three times the output of product A. The storage space
        required for product A is four times that of product B, and a maximum
        of four kilograms of product A can be stored per week. Formulate a
        linear programming model for this problem.
        """
    )

    assert result.matched is True
    assert result.template_id == "narrative_product_mix_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 146.66666666666669)
    assert math.isclose((result.variable_values or {})["produce_A"], 4 / 3)


def test_template_solver_handles_weighted_idle_goal_product_mix():
    result = solve_with_template(
        """
        A company produces two types of small motorcycles. Type A is entirely
        manufactured in-house, while Type B is assembled from imported parts.

        | Type | Manufacturing (hours/unit) | Assembly (hours/unit) | Inspection (hours/unit) | Selling Price (Yuan/unit) |
        | :---: | :---: | :---: | :---: | :---: |
        | Type A | 20 | 5 | 3 | 650 |
        | Type B | 0 | 7 | 6 | 725 |
        | Max weekly capacity | 120 | 80 | 40 | - |
        | Process cost (Yuan/hour) | 12 | 8 | 10 | - |

        p_1: The total weekly profit should be at least 3000 yuan.
        p_2: At least 5 units of Type A must be produced each week.
        p_3: Idle time of each process should be minimized, with weights
        proportional to the corresponding hourly cost. Overtime is not allowed.

        What is the total profit of the company's production plan?
        """
    )

    assert result.matched is True
    assert result.template_id == "weighted_idle_goal_product_mix_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 3867)
    assert result.variable_values == {"produce_Type A": 6.0, "produce_Type B": 3.0}


def test_template_solver_handles_minimum_overtime_production_goal():
    result = solve_with_template(
        """
        A textile factory produces clothing fabric and curtain fabric. The
        weekly production time is set at 110 hours. Both fabrics are produced
        at a rate of 1000 meters per hour. At least 70,000 meters of curtain
        fabric and at least 45,000 meters of clothing fabric must be sold.

        p_1: The weekly production time must fully utilize 110 hours;
        p_2: Overtime should not exceed 10 hours per week;
        p_3: The stated fabric sales targets must be met;
        p_4: Minimize overtime as much as possible.
        """
    )

    assert result.matched is True
    assert result.template_id == "minimum_overtime_production_goal"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 5)


def test_template_solver_handles_sales_staff_overtime_goal():
    result = solve_with_template(
        """
        A shoe store employs 5 full-time sales clerks and 4 part-time sales
        clerks.

        |  | Monthly Working Hours | Sales Volume (Pairs/Hour) | Wage (Yuan/Hour) | Overtime Pay (Yuan/Hour) |
        | :---: | :---: | :---: | :---: | :---: |
        | Full-time | 160 | 5 | 1 | 1.5 |
        | Part-time | 80 | 2 | 0.6 | 0.7 |

        p_1: Achieve monthly sales of 5500 pairs;
        p_2: Ensure full employment of all sales clerks;
        p_3: Minimize overtime hours.
        """
    )

    assert result.matched is True
    assert result.template_id == "sales_staff_overtime_goal"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 172)
    assert result.variable_values and math.isclose(
        result.variable_values["overtime_hours_Full-time"], 172
    )


def test_template_solver_handles_continuous_overtime_resource_product_mix():
    result = solve_with_template(
        """
        A company produces two kinds of products. A product of the first type
        requires 1/4 hours of assembly labor, 1/8 hours of testing, and $1.2
        worth of raw materials. A product of the second type requires 1/3
        hours of assembly, 1/3 hours of testing, and $0.9 worth of raw
        materials. Given the current personnel, there can be at most 90 hours
        of assembly labor and 80 hours of testing each day. Suppose that up
        to 50 hours of overtime assembly labor can be scheduled, at a cost of
        $7 per hour. Products of the first and second type have a market value
        of $9 and $8 respectively. Try to maximize daily profit. Provide your
        answer rounded to the nearest dollar.
        """
    )

    assert result.matched is True
    assert result.template_id == "overtime_resource_product_mix"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 4018)
    assert result.variable_values and math.isclose(
        result.variable_values["overtime_assembly_hours"], 50
    )


def test_template_solver_handles_integer_overtime_resource_product_mix():
    result = solve_with_template(
        """
        A company uses steel and aluminum as raw materials to produce two
        products, A and B. A single unit of product A requires 6 kg of steel,
        8 kg of aluminum, 11 hours of labor, and yields a profit of 5000 yuan
        excluding worker overtime pay. A single unit of product B requires
        12 kg of steel, 20 kg of aluminum, 24 hours of labor, and yields a
        profit of 11000 yuan excluding worker overtime pay. The company
        currently has 200 kg of steel, 300 kg of aluminum, and 300 hours of
        labor available. If workers need to work overtime, the overtime pay is
        100 yuan per hour. Maximize profit with minimal worker overtime.
        """
    )

    assert result.matched is True
    assert result.template_id == "overtime_resource_product_mix"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 165900)
    assert result.artifact and result.artifact["integer_products"] is True


def test_template_solver_handles_livestock_resource_mix_ilp():
    result = solve_with_template(
        """
        A farmer needs to decide how many cows, sheep, and chickens to raise
        in order to achieve maximum profit. The farmer can sell cows, sheep,
        and chickens for $500, $200, and $8 each, respectively. The feed costs
        for each cow, sheep, and chicken are $100, $80, and $5, respectively.
        Each cow, sheep, and chicken produces 10, 5, and 3 units of manure per
        day, respectively. The staff can handle up to 800 units of manure. The
        farmer can raise at most 50 chickens, must have at least 10 cows, and
        must also raise at least 20 sheep. Finally, the total number of animals
        cannot exceed 100.
        """
    )

    assert result.matched is True
    assert result.template_id == "livestock_resource_mix_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 30400)


def test_template_solver_handles_two_product_time_ratio_lp():
    result = solve_with_template(
        """
        A company produces two products (A and B), with a profit of £3 and £5
        per unit sold, respectively. Product A requires 12 minutes of assembly
        time per unit for product A and 25 minutes per unit for product B.
        Effective machine working time per week is only 30 hours. For every
        five units of product A produced, at least two units of product B must
        be produced.
        """
    )

    assert result.matched is True
    assert result.template_id == "two_product_time_ratio_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 409.0909, rel_tol=1e-5)


def test_template_solver_handles_production_conversion_lp():
    result = solve_with_template(
        r"""
        A dairy processing plant uses milk to produce two dairy products,
        \( A_{1} \) and \( A_{2} \). One barrel of milk can be processed into
        3 kg of \( A_{1} \) in 12 hours on Type A equipment or into 4 kg of
        \( A_{2} \) in 8 hours on Type B equipment. The profit is 24 yuan per
        kilogram of \( A_{1} \) and 16 yuan per kilogram of \( A_{2} \). The
        processing plant can get a daily supply of 50 barrels of milk, with a
        total of 480 hours of labor time available from regular workers each
        day. The Type A equipment can process up to 100 kg of \( A_{1} \) per
        day, while the processing capacity of Type B equipment is not limited.
        Formulate a production plan for the plant to maximize daily profit.
        """
    )

    assert result.matched is True
    assert result.template_id == "production_conversion_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 3360)


def test_template_solver_handles_quality_constrained_blending_lp():
    result = solve_with_template(
        r"""
        A company mixes four types of liquid raw materials with different
        sulfur contents, denoted as A, B, C, and D, to produce two products,
        denoted as \( \mathrm{A} \) and \( \mathrm{B} \). The sulfur contents
        of raw materials A, B, C, and D are \( 3\%, 1\%, 2\%, 1\% \)
        respectively, and their purchase prices are 6, 16, 10, 15 thousand
        yuan per ton respectively. The sulfur content of products
        \( \mathrm{A} \) and \( \mathrm{B} \) must not exceed
        \( 2.5\% \) and \( 1.5\% \) respectively, and their selling prices
        are 9.15 thousand yuan per ton. There is no limit to the supply of raw
        materials A, B, and C, but the supply of raw material D is limited to a
        maximum of 50 tons. The market demand for products \( \mathrm{A} \)
        and \( \mathrm{B} \) is 100 tons and 200 tons respectively. How should
        production be arranged to maximize the profit?
        """
    )

    assert result.matched is True
    assert result.template_id == "quality_constrained_blending_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 115)
    assert result.variable_values == {
        "blend_raw_A_to_product_A": 50.0,
        "blend_raw_C_to_product_A": 50.0,
        "produce_A": 100.0,
    }


def test_template_solver_handles_gasoline_blending_lp():
    result = solve_with_template(
        """
        Solar Oil Company sells regular and premium gasoline. It makes these
        products by blending four raw gasolines. Raw gasoline 1 with 86 octane
        is available up to 20,000 barrels per day at a cost of $17.00 per
        barrel, raw gasoline 2 with 88 octane can be purchased up to 15,000
        barrels per day at $18.00 per barrel, raw gasoline 3 with 92 octane is
        available up to 15,000 barrels daily at $20.50 per barrel, and raw
        gasoline 4 with 96 octane has a daily availability of 10,000 barrels
        at $23.00 per barrel. The required minimum octane for each final
        gasoline product is 89 for regular, which sells at $19.50 per barrel
        with a maximum daily demand of 35,000 barrels, and 93 for premium
        gasoline, priced at $22.00 per barrel with a demand of up to 23,000
        barrels per day. The blending of gasoline is linear in volume and
        octane. Find the maximal profit.
        """
    )

    assert result.matched is True
    assert result.template_id == "gasoline_blending_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 42142.86, abs_tol=0.01)


def test_template_solver_handles_consecutive_day_workforce_scheduling():
    result = solve_with_template(
        """
        A post office requires different numbers of full-time employees on
        different days of the week. Monday requires 17 employees, Tuesday
        requires13, Wednesday requires 15, Thursday requires 19, Friday
        requires 14, Saturday requires 16, Sunday requires 11. Union rules
        state that each full-time employee must work five consecutive days and
        then receive two days off. The post office wants to minimize the
        number of full-time employees who must be hired.
        """
    )

    assert result.matched is True
    assert result.template_id == "consecutive_day_workforce_scheduling"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 23)


def test_template_solver_handles_cyclic_shift_staffing_inline_requirements():
    result = solve_with_template(
        """
        The number of salespeople required at a 24-hour convenience store in
        different time periods is as follows: 2:00-6:00 - 10 people,
        6:00-10:00 - 15 people, 10:00-14:00 - 25 people, 14:00-18:00 - 20
        people, 18:00-22:00 - 18 people, 22:00-2:00 - 12 people.
        Salespeople start their shifts at 2:00, 6:00, 10:00, 14:00, 18:00,
        and 22:00, working continuously for 8 hours. Determine the minimum
        number of salespeople needed to meet the requirements.
        """
    )

    assert result.matched is True
    assert result.template_id == "cyclic_shift_staffing"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 53)


def test_template_solver_handles_cyclic_shift_staffing_markdown_table():
    result = solve_with_template(
        """
        A restaurant operates around the clock, and the number of waiters
        needed in 24 hours is shown in the table.

        | Time | Minimum Number of Waiters Needed | Time | Minimum Number of Waiters Needed |
        |:-----:|:-------------------------------:|:-----:|:-------------------------------:|
        | $2 \\sim 6$ | 4 | $14 \\sim 18$ | 7 |
        | $6 \\sim 10$ | 8 | $18 \\sim 22$ | 12 |
        | $10 \\sim 14$ | 10 | $22 \\sim 2$ | 4 |

        Each waiter works continuously for 8 hours a day. The goal is to find
        the minimum number of waiters.
        """
    )

    assert result.matched is True
    assert result.template_id == "cyclic_shift_staffing"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 26)


def test_template_solver_handles_cyclic_shift_staffing_pay_mode():
    result = solve_with_template(
        """
        The number of nurses required in each time period over 24 hours is as
        follows: 2:00-6:00 - 10 people, 6:00-10:00 - 15 people,
        10:00-14:00 - 25 people, 14:00-18:00 - 20 people, 18:00-22:00 - 18
        people, 22:00-2:00 - 12 people. Nurses start shifts in 6 batches and
        work continuously for 8 hours. The pay for regular nurses is
        10 yuan/hour and for contract nurses is 15 yuan/hour. Determine the
        minimum pay.
        """
    )

    assert result.matched is True
    assert result.template_id == "cyclic_shift_staffing"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 4240)
    assert result.artifact and result.artifact["wage_type_used"] == "regular"


def test_template_solver_handles_relaxed_consecutive_day_workforce_scheduling():
    result = solve_with_template(
        """
        A hospital wants to make a weekly night shift schedule for its nurses.
        The demand for the night shift on day j is d_j for j = 1, ..., 7.
        d1 = 5, d2 = 4, d3 = 7, d4 = 3, d5 = 8, d6 = 4, d7 = 3. Every nurse
        works 5 days in a row. We want to minimize the total number of nurses
        used while meeting all demand. Ignore the integrality constraints for
        now, so half nurses are allowed. Provide your answer rounded to the
        nearest integer.
        """
    )

    assert result.matched is True
    assert result.template_id == "consecutive_day_workforce_scheduling"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 8)
    assert result.artifact and result.artifact["relax_integrality"] is True


def test_template_solver_handles_student_duty_scheduling_milp():
    result = solve_with_template(
        """
        A university computer lab hires 4 undergraduates (designated 1, 2, 3,
        and 4) and 2 graduate students (designated 5 and 6) for duty answering
        questions. The maximum duty hours from Monday to Friday and the hourly
        wage for each person are shown below.

        | Student ID | Wage (CNY/h) | Monday | Tuesday | Wednesday | Thursday | Friday |
        |------------|--------------|--------|---------|-----------|----------|--------|
        | 1          | 10.0         | 6      | 0       | 6         | 0        | 7      |
        | 2          | 10.0         | 0      | 8       | 9         | 6        | 0      |
        | 3          | 9.9          | 4      | 8       | 3         | 0        | 5      |
        | 4          | 9.8          | 5      | 5       | 6         | 0        | 4      |
        | 5          | 10.8         | 3      | 0       | 5         | 8        | 0      |
        | 6          | 11.3         | 0      | 6       | 0         | 6        | 5      |

        The lab operates from 8:00 AM to 10:00 PM, and there must be one and
        only one student on duty during open hours. Each undergraduate must
        work at least 8 hours per week, and each graduate student must work at
        least 7 hours per week. Each student can work no more than 2 shifts per
        week, and no more than 3 students can be scheduled for duty each day.
        Minimize gross pay.
        """
    )

    assert result.matched is True
    assert result.template_id == "student_duty_scheduling_milp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 715.7)


def test_template_solver_handles_fractional_investment_budget_lp():
    result = solve_with_template(
        """
        Star Oil Company is considering five different investment opportunities.
        Investment 1 has a time 0 cash outflow of $11 million and a time 1 cash
        outflow of $3 million, with an NPV of $13 million. Investment 2
        requires a time 0 outflow of $53 million and a time 1 cash outflow of
        $6 million, yielding an NPV of $16 million. Investment 3 has smaller
        outflows of $5 million at both time 0 and time 1 and also an NPV of
        $16 million. Investment 4 asks for a time 0 outflow of $5 million and
        a time 1 cash outflow of $1 million, with an NPV of $14 million.
        Investment 5 requires $29 million outflow at time 0 and a significant
        $34 million at time 1, resulting in an NPV of $39 million. Star
        Oil has $40 million available for investment at time 0, and it
        estimates that $20 million will be available for investment at time 1.
        Star Oil may purchase any fraction of each investment. Maximize NPV.
        """
    )

    assert result.matched is True
    assert result.template_id == "fractional_investment_budget_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 57.449, abs_tol=0.001)


def test_template_solver_handles_reinvestable_cashflow_lp():
    result = solve_with_template(
        """
        You have an initial capital of 500,000 yuan at the beginning of Year 1.
        Over the next three years, you may allocate funds to the following
        investment projects. The goal is to choose an investment plan that
        maximizes the total principal plus interest at the end of Year 3. No
        borrowing is allowed, and you may reallocate available cash at each
        decision time.

        Projects: (1) A 1-year product available at the beginning of each year.
        If you invest I at the start of a year, you receive 1.20*I at that
        year's end. Unlimited capacity. (2) A 2-year product available only at
        the beginning of Year 1. It matures at the end of Year 2 and pays
        1.50*I. Investment in this product is capped at 120,000 yuan. (3) A
        same-year product available at the beginning of Year 2, maturing at the
        end of Year 2, and paying 1.60*I. Investment is capped at 150,000 yuan.
        (4) A 1-year product available at the beginning of Year 3, maturing at
        the end of Year 3, and paying 1.40*I. Investment is capped at 100,000
        yuan. Decisions are made at the beginnings of Years 1-3 using only
        currently available cash.
        """
    )

    assert result.matched is True
    assert result.template_id == "reinvestable_cashflow_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 964640)


def test_template_solver_handles_reinvestable_return_wording():
    result = solve_with_template(
        """
        An investor plans to invest 100,000 yuan, with two investment options
        to choose from. The first investment guarantees a return of 0.7 yuan
        for every 1 yuan invested after one year. The second investment
        guarantees a return of 2 yuan for every 1 yuan invested after two
        years, but the investment time must be in multiples of two years. In
        order to maximize the investor's earnings by the end of the third
        year, how should the investments be made?
        """
    )

    assert result.matched is True
    assert result.template_id == "reinvestable_cashflow_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 510000)


def test_template_solver_handles_reinvestable_principal_interest_limits():
    result = solve_with_template(
        """
        Someone has a fund of 300,000 yuan and has the following investment
        projects in the next three years: (1) Investment can be made at the
        beginning of each year within three years, with an annual profit of 20%
        of the investment amount, and the principal and interest can be used
        for investment in the following year; (2) Investment is only allowed at
        the beginning of the first year, and it can be recovered at the end of
        the second year, with the total principal and interest amounting to
        150% of the investment amount, but the investment limit is no more than
        150,000 yuan; (3) Investment is allowed at the beginning of the second
        year within three years, and it can be recovered at the end of the third
        year, with the total principal and interest amounting to 160% of the
        investment amount, and the investment limit is 200,000 yuan; (4)
        Investment is allowed at the beginning of the third year within three
        years, and it can be recovered in one year with a profit of 40%, and
        the investment limit is 100,000 yuan. Maximize the principal and
        interest at the end of the third year.
        """
    )

    assert result.matched is True
    assert result.template_id == "reinvestable_cashflow_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 580000)


def test_template_solver_handles_balance_sheet_production_lp():
    result = solve_with_template(
        """
        Semicond manufactures tape recorders and radios. The per-unit labor
        costs for tape recorders and radios are $50 and $35 respectively, and
        the raw material costs are $30 for tape recorders and $40 for radios.
        The selling prices are $100 for a tape recorder and $90 for a radio.
        On December 1, Semicond has enough raw material to manufacture 100 tape
        recorders and 100 radios. The balance sheet shows cash at $10,000,
        accounts receivable at $3,000, inventory outstanding valued at $7,000,
        and a bank loan liability of $10,000. All sales in December are on
        credit and payment is not received until February. Semicond will
        collect $2,000 in accounts receivable in December, pay off $1,000 of
        its loan, and pay monthly rent of $1,000. On January 1, it receives raw
        materials worth $2,000, to be paid for in February. Management requires
        a minimum cash balance of $4,000 and a current ratio of at least 2.
        Find the maximal contribution to profit from December's production.
        """
    )

    assert result.matched is True
    assert result.template_id == "balance_sheet_production_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 2500)


def test_template_solver_handles_binary_purchase_selection_knapsack():
    result = solve_with_template(
        """
        A family is deciding whether to purchase several properties. The
        annual income from Property 1 is $12,500, Property 2 is $35,000,
        Property 3 is $23,000, and Property 4 is $100,000. The decision to be
        made is whether to buy each property or not, as there is only one of
        each property available. Help them decide which properties to purchase
        to maximize their annual income.

        The cost of Property 1 is $1.5 million, Property 2 is $2.1 million,
        Property 3 is $2.3 million, and Property 4 is $4.2 million. The
        investment budget is $7 million.

        If they purchase Property 4, they cannot purchase Property 3.
        """
    )

    assert result.matched is True
    assert result.template_id == "binary_purchase_selection_knapsack"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 135000)


def test_template_solver_handles_binary_child_trip_selection():
    result = solve_with_template(
        """
        The Li family has 5 children: Alice, Bob, Charlie, Diana, and Ella.
        The cost to take Alice is $1000, Bob is $900, Charlie is $600, Diana
        is $500, and Ella is $700. Which children should the couple take to
        minimize the total cost? They can take up to 3 children. Bob is the
        youngest, so the family will definitely take him. If the couple takes
        Alice, they will not take Diana. If the couple takes Bob, they will
        not take Charlie. If they take Charlie, they must also take Diana. If
        they take Diana, they must also take Ella. The family has decided to
        take at least two children.
        """
    )

    assert result.matched is True
    assert result.template_id == "binary_subset_selection_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 1600)
    assert result.artifact and result.artifact["selected"] == ["Bob", "Ella"]


def test_template_solver_handles_binary_candidate_hiring_selection():
    result = solve_with_template(
        """
        A company hopes to recruit new employees. The salary requirements for
        candidates A, B, C, D, and E are $8100, $20000, $21000, $3000, and
        $8000 respectively. They want to minimize the total amount paid, hire
        a maximum of 3 new employees, and stay within a budget of $35000.
        Candidate A: Bachelor's degree; Candidate B: Master's degree;
        Candidate C: Doctoral degree; Candidate D: No degree; Candidate E:
        No degree. They will select at least one candidate with a Master's or
        Doctoral degree. Candidate A: 3 years of work experience; Candidate B:
        10 years of work experience; Candidate C: 4 years of work experience;
        Candidate D: 3 years of work experience; Candidate E: 7 years of work
        experience. The total work experience must be no less than 12 years.
        Due to equivalent professional skills of candidates A and E, choose at
        most one from the two. They will hire at least 2 new employees.
        """
    )

    assert result.matched is True
    assert result.template_id == "binary_subset_selection_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 23000)
    assert result.artifact and result.artifact["selected"] == ["B", "D"]


def test_template_solver_handles_candy_quality_blending_table():
    result = solve_with_template(
        """
        A candy factory uses raw materials A, B, and C to process three
        different brands of candies, A, B, and C.

        | Item | A | B | C | Raw Material Cost (Yuan/kg) | Monthly Limit (kg) |
        |------|---|---|---|-----------------------------|--------------------|
        | A | >= 60% | >= 15% | | 2.00 | 2000 |
        | B | | | | 1.50 | 2500 |
        | C | <= 20% | <= 60% | <= 50% | 1.00 | 1200 |
        | Processing Fee (Yuan/kg) | 0.50 | 0.40 | 0.30 | | |
        | Selling Price (Yuan/kg) | 3.40 | 2.85 | 2.25 | | |

        How many kilograms of each brand should be produced to maximize
        profit?
        """
    )

    assert result.matched is True
    assert result.template_id == "candy_quality_blending_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 6160)


def test_template_solver_handles_two_product_seasonal_inventory_plan():
    result = solve_with_template(
        """
        The market demand for products I and II is as follows: Product I
        requires 10,000 units per month from January to April, 30,000 units
        per month from May to September, and 100,000 units per month from
        October to December. Product II requires 15,000 units per month from
        March to September and 50,000 units per month during other months.
        Product I costs 5 yuan per unit to produce from January to May, and
        4.50 yuan per unit from June to December; Product II costs 8 yuan per
        unit to produce from January to May, and 7 yuan per unit from June to
        December. The factory's combined production capacity for both products
        should not exceed 120,000 units per month. Product I has a volume of
        0.2 cubic meters per unit, Product II has a volume of 0.4 cubic meters
        per unit, and the factory's warehouse capacity is 15,000 cubic meters.
        Using the factory's own warehouse costs 1 yuan per cubic meter per
        month, while renting an external warehouse increases this cost to 1.5
        yuan per cubic meter per month. Given that the initial inventory of
        both products at the beginning of July is zero, schedule production
        from July to December to minimize total production and inventory costs.
        """
    )

    assert result.matched is True
    assert result.template_id == "two_product_seasonal_inventory_plan_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 3160500)


def test_template_solver_handles_route_vehicle_min_cost():
    result = solve_with_template(
        """
        A transportation manager is planning to allocate vehicles among three
        different routes: X, Y, and Z. The objective is to minimize the total
        operating cost, with costs being $5, $7$, and $3$ per vehicle for
        routes X, Y, and Z respectively. The combined number of vehicles on
        routes X, Y, and Z cannot exceed 1000. Twice the number of vehicles on
        route X plus thrice the number of vehicles on route Y should be at
        least 500. The sum of vehicles on routes X and Z minus those on route
        Y should not exceed 400. The number of vehicles running on route Y
        should be at least 100 more than those on route Z. Allocations must be
        whole numbers.
        """
    )

    assert result.matched is True
    assert result.template_id == "route_vehicle_min_cost_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 1167)


def test_template_solver_handles_healthcare_fund_three_department_min_cost():
    result = solve_with_template(
        """
        In a healthcare scenario, a hospital administrator is planning to
        allocate funds across three departments: X (General Medicine), Y
        (Pediatrics), and Z (Surgery). These allocations need to be whole
        numbers. The total budget for all three departments combined cannot
        exceed $1000. Department X requires an allocation that is at least
        $200 more than twice the allocation for department Y, while department
        Z requires an allocation that exceeds the allocation for department Y
        by at least $150. Returns are quantified as 50 units for department X,
        30 units for department Y, and 20 units for department Z. Minimize the
        total cost.
        """
    )

    assert result.matched is True
    assert result.template_id == "healthcare_fund_three_department_min_cost_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 13000)


def test_template_solver_handles_military_support_points_min_cost():
    result = solve_with_template(
        """
        A military commander is planning to allocate resources between four
        types of units, X1, X2, X3 and X4. The allocations must be whole
        numbers. The total number of units that can be supported for X1 and
        X2 combined is limited to 200. The combined strength of twice the
        units of X1 and three and a half times the units of X3 must be at
        least 400. The difference in units between X2 and half of those
        allocated to unit type X4 cannot exceed 50. After subtracting the
        number of units for both unit types X1 and unit type X3 from those for
        unit type x2, it should not exceed 70. Unit type x1 needs 10000
        points, Unit type x2 needs 5000 points, Unit type x3 needs 8000 points
        while Unit type x4 requires 6000 points. The allocation for each unit
        cannot exceed its maximum capacity (150 for x1, 100 for x2, 80 for x3,
        20 for x4). Minimize support points.
        """
    )

    assert result.matched is True
    assert result.template_id == "military_support_points_min_cost_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 1240000)


def test_template_solver_handles_two_pill_nutrition_min_mix():
    result = solve_with_template(
        """
        A pharmacy has 3000 mg of morphine to make painkillers and sleeping
        pills. Each painkiller pill requires 10 mg of morphine and 3 units of
        digestive medicine. Each sleeping pill requires 6 mg of morphine and
        5 units of digestive medicine. The pharmacy needs to make at least 50
        painkiller pills. At least 70% of the pills should be sleeping pills.
        Minimize the total amount of digestive medicine needed.
        """
    )

    assert result.matched is True
    assert result.template_id == "two_item_nutrition_min_mix_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 735)


def test_template_solver_handles_burger_pizza_cholesterol_min_mix():
    result = solve_with_template(
        """
        A doctor recommends that a man eat burgers and pizza. Each burger
        contains 10 units of fat and 300 calories. Each slice of pizza
        contains 8 units of fat and 250 calories. The man must get at least
        130 units of fat and 3000 calories. Each burger contains 12 units of
        cholesterol while each slice of pizza contains 10 units of
        cholesterol. He eats at least twice as many slices of pizza as
        burgers. Minimize cholesterol intake.
        """
    )

    assert result.matched is True
    assert result.template_id == "two_item_nutrition_min_mix_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 160)


def test_template_solver_handles_two_volunteer_gift_max_mix():
    result = solve_with_template(
        """
        A toy store hires seasonal and full-time volunteers to deliver gifts.
        A seasonal volunteer can deliver 5 gifts and gets 2 points. A
        full-time volunteer can deliver 8 gifts and gets 5 points. The store
        can only give out 200 points. A maximum of 30% of the volunteers can
        be seasonal and at least 10 must be full-time. Maximize the total
        number of gifts that can be delivered.
        """
    )

    assert result.matched is True
    assert result.template_id == "two_volunteer_gift_max_mix_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 342)


def test_template_solver_handles_two_asset_real_estate_profit():
    result = solve_with_template(
        """
        My family has decided to invest in real state. Currently, they have
        $600,000 to invest, some in apartments and the rest in townhouses. The
        money invested in apartments must not be greater than $200,000. The
        money invested in apartments must be at least a half as much as that
        in townhouses. If the apartments earn 10%, and the townhouses earn
        15%, maximize profit.
        """
    )

    assert result.matched is True
    assert result.template_id == "two_asset_real_estate_profit_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 80000)


def test_template_solver_handles_salmon_eggs_sodium_min_mix():
    result = solve_with_template(
        """
        A macro-counting fitness guru only eats salmon and eggs. Each bowl of
        salmon contains 300 calories, 15 grams of protein, and 80 mg of sodium.
        Each bowl of eggs contains 200 calories, 8 grams of protein, and 20 mg
        of sodium. At most 40% of his meals can be eggs. He needs to eat at
        least 2000 calories and 90 grams of protein. Minimize sodium intake.
        """
    )

    assert result.matched is True
    assert result.template_id == "two_item_nutrition_min_mix_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 430.7692307692307)


def test_template_solver_handles_container_loading_min_count():
    result = solve_with_template(
        """
        Each container can hold a maximum of 60 tons of goods and each
        container used must load at least 18 tons of goods. Goods include five
        types: A, B, C, D, and E, with quantities of 120, 90, 300, 90, and
        120 respectively. The weights are 0.5 tons for A, 1 ton for B, 0.4
        tons for C, 0.6 tons for D, and 0.65 tons for E. Every time A goods
        are loaded, at least 1 unit of C must also be loaded, and each
        container must load at least 12 units of D. Use the fewest number of
        containers.
        """
    )

    assert result.matched is True
    assert result.template_id == "container_loading_min_count_milp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 7)


def test_template_solver_handles_input_output_gdp_lp():
    result = solve_with_template(
        """
        Carelland exports steel, engines, electronic components, and plastic.
        The unit prices of steel, engines, electronics, and plastic on the
        world market are 500, 1500, 300, 1200 respectively. Producing 1 unit
        of steel requires 0.02 units of engines, 0.01 units of plastic, 250
        Klunz of other imported goods, and 6 person-months of labor. Producing
        1 unit of engines requires 0.8 units of steel, 0.15 units of
        electronic components, 0.11 units of plastic, 300 Klunz of imported
        goods, and 1 person-year. One unit of electronics requires: 0.01 units
        of steel, 0.01 units of engines, 0.05 units of plastic, 50 Klunz of
        imported goods, and 6 person-months of labor. One unit of plastic
        requires: 0.03 units of engines, 0.2 units of steel, 0.05 units of
        electronic components, 300 Klunz of imported goods, and 2 person-years.
        Engine production is limited to 650000 units, and plastic production
        is limited to 60000 units. The total available labor force per year is
        830000 person-months. Maximize domestic GDP.
        """
    )

    assert result.matched is True
    assert result.template_id == "input_output_gdp_lp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 43090833.43, rel_tol=0.05)


def test_template_solver_handles_two_exercise_balance_min_fatigue():
    result = solve_with_template(
        """
        A sports coach is planning to allocate training hours between two
        exercises. The total number of hours for both exercises combined is
        limited to 10. The effectiveness score is three times the hours spent
        on exercise X plus four times those spent on exercise Y, and must be
        at least 30. The difference in hours between exercise X and Y should
        not exceed 2. Fatigue scores are 40 for exercise X and 60 for exercise
        Y. X and Y are integers, and the coach cannot spend more than six
        hours on exercise X or eight hours on exercise Y. Minimize fatigue.
        """
    )

    assert result.matched is True
    assert result.template_id == "two_exercise_balance_min_fatigue_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 440)


def test_template_solver_handles_telecom_project_pair_min_cost():
    result = solve_with_template(
        """
        A telecommunications company is planning resources between four
        projects: $x1$, $x2$, $x3$ and $x4$. The cost associated with each
        project is 50, 75, 100, and 125 units per resource for $x1$, $x2$,
        $x3$ and $x4$ respectively. The total resources allocated to the first
        two projects ($x1$ and $x2$) cannot exceed 5000 units. The total
        resources allocated to the last two projects ($x3$ and $x4$) are
        capped at 3000 units. At least 2000 units must be devoted between
        project $x1$ and project $x3$. A minimum of 4000 units needs to be
        shared between project$x2$ and project$x4$. Integer allocations are
        required. Minimize total cost.
        """
    )

    assert result.matched is True
    assert result.template_id == "telecom_project_pair_min_cost_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 450000)


def test_template_solver_handles_fractional_telecom_project_min_cost():
    result = solve_with_template(
        """
        A telecommunications company is planning resources across projects X1,
        X2, X3, and X4. Each project has cost per unit: 1000 for project X1,
        2000 for project X2, 3000 for project X3 and 4000 for project X4.
        Projects X1 and X2 cannot exceed 500 units. Projects X3 and X4 cannot
        exceed 800 units. Half of the resources allocated to project X1 must
        be at least one-fourth more than those allocated to Project X3 by no
        less than 50 units. The difference between three-quarters of the
        resources allocated to Project X4 and those assigned to Project X2
        must be at least 100 units. For Project x1: between 0 and 500 units,
        For Project x2: between 0 and 400 units, For Project x3: between 0
        and 600 units, For Project x4: between 0 and 700 units. Integer
        allocations are required. Minimize cost.
        """
    )

    assert result.matched is True
    assert result.template_id == "fractional_telecom_project_min_cost_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 636000)


def test_template_solver_handles_four_property_real_estate_infeasible():
    result = solve_with_template(
        """
        A real estate developer invests in residential (x1), commercial (x2),
        industrial (x3), and retail (x4). Costs are $200,000, $300,000,
        $500,000 and $700,000 for each unit of x1, x2, x3 and x4 respectively.
        Residential and commercial properties cannot exceed 50. Twice the
        number of industrial properties plus retail properties should be at
        least 60. Three times the number of residential properties plus four
        point five times commercial properties should not exceed 100. Six
        times the number of industrial properties minus half a unit of retail
        property should not surpass 80. Residential(x1): Up to 20 units.
        Commercial(x2): Up to 30 units. Industrial(x3): Up to 15 units.
        Retail(x4): Up to 10 units. All investments must be whole numbers.
        Minimize cost.
        """
    )

    assert result.matched is True
    assert result.template_id == "four_property_real_estate_min_cost_ilp"
    assert result.status == "infeasible"


def test_template_solver_handles_wrap_platter_time_min():
    result = solve_with_template(
        """
        A fast food place makes wraps and platters. Each wrap requires 5 units
        of meat and 3 units of rice. Each platter requires 7 units of meant
        and 5 units of rice. Each wrap takes 10 minutes to make, each platter
        takes 8 minutes to make. The fast food place must use at least 3000
        units of meat and 2500 units of rice. At least 3 times as many wraps
        need to be made as platter. Minimize total production time.
        """
    )

    assert result.matched is True
    assert result.template_id == "wrap_platter_time_min_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 6794)


def test_template_solver_handles_two_vehicle_capacity_min_count():
    laundromat = solve_with_template(
        """
        A laundromat can buy top-loading and front-loading washing machines.
        The top-loading model can wash 50 items per day while the front-loading
        model can wash 75 items per day. The top-loading model consumes 85 kWh
        per day while the front-loading model consumes 100 kWh per day. The
        laundromat must wash at least 5000 items per day and has available
        7000 kWh per day. At most 40% of the machines can be top-loading. At
        least 10 machines should be front-loading. Minimize the total number
        of washing machines.
        """
    )
    buses = solve_with_template(
        """
        A school field trip needs small buses and large buses. A small bus can
        carry 20 students while a large bus can carry 50 students. The school
        needs transportation for at least 500 students. A maximum of 20% of
        the buses can be large buses. Minimize the total number of buses.
        """
    )

    assert laundromat.matched is True
    assert laundromat.template_id == "two_vehicle_capacity_min_count_ilp"
    assert laundromat.status == "optimal"
    assert math.isclose(laundromat.objective_value or 0, 67)
    assert buses.matched is True
    assert buses.template_id == "two_vehicle_capacity_min_count_ilp"
    assert buses.status == "optimal"
    assert math.isclose(buses.objective_value or 0, 20)


def test_template_solver_handles_two_product_resource_profit_variants():
    bakery = solve_with_template(
        """
        A bakery bakes bagels and croissants. A batch of bagels can be made
        using 2 hours of oven time and 0.25 hours of pastry chef time.
        Croissants take 1 hour of oven time, they take 2 hours of pastry chef
        time. In a day, the bakery has at most 70 hours available for the oven
        and 32 pastry chef hours available. The profit per batch is $20 and
        $40 respectively. Maximize profit.
        """
    )
    farmer = solve_with_template(
        """
        A farmer has 500 acres of land to grow turnips and pumpkins. Turnips
        require 50 minutes of watering and $80 worth of pesticide. Pumpkins
        require 90 minutes of watering and $50 worth of pesticide. The farmer
        has 40000 minutes available for watering and $34000 available to spend
        on pesticide. The revenue per acre of turnips is $300 and the revenue
        per acre of pumpkins is $450. Maximize revenue.
        """
    )

    assert bakery.matched is True
    assert bakery.template_id == "two_product_resource_profit_max_ilp"
    assert bakery.status == "optimal"
    assert math.isclose(bakery.objective_value or 0, 1060)
    assert farmer.matched is True
    assert farmer.template_id == "two_product_resource_profit_max_lp"
    assert farmer.status == "optimal"
    assert math.isclose(farmer.objective_value or 0, 206250)


def test_template_solver_handles_vrp_hard_time_windows_milp():
    result = solve_with_template(
        """
        Vehicle Routing Problem with Hard Time Windows. The company can use at
        most 5 trucks, and the capacity of each truck is 200 units. Minimize
        the total distance traveled by all vehicles.

        Central Depot (Depot 0):
        Coordinates: (40, 50)
        Operating Time Window: [0, 1236] (minutes)

        | Customer ID | Coordinates (X, Y) | Demand (units) | Time Window (minutes) | Service Duration (minutes) |
        |-------------|--------------------|----------------|------------------------|----------------------------|
        | 1 | (45, 68) | 10 | [912, 967] | 90 |
        | 2 | (45, 70) | 30 | [825, 870] | 90 |
        | 3 | (42, 66) | 10 | [65, 146] | 90 |
        | 4 | (42, 68) | 10 | [727, 782] | 90 |
        | 5 | (42, 65) | 10 | [15, 67] | 90 |
        | 6 | (40, 69) | 20 | [621, 702] | 90 |
        | 7 | (40, 66) | 20 | [170, 225] | 90 |
        | 8 | (38, 68) | 20 | [255, 324] | 90 |
        | 9 | (38, 70) | 10 | [534, 605] | 90 |
        | 10 | (35, 66) | 10 | [357, 410] | 90 |
        | 11 | (35, 69) | 10 | [448, 505] | 90 |
        | 12 | (25, 85) | 20 | [652, 721] | 90 |
        | 13 | (22, 75) | 30 | [30, 92] | 90 |
        | 14 | (22, 85) | 10 | [567, 620] | 90 |
        | 15 | (20, 80) | 40 | [384, 429] | 90 |
        | 16 | (20, 85) | 40 | [475, 528] | 90 |
        | 17 | (18, 75) | 20 | [99, 148] | 90 |
        | 18 | (15, 75) | 20 | [179, 254] | 90 |
        | 19 | (15, 80) | 10 | [278, 345] | 90 |
        | 20 | (30, 50) | 10 | [10, 73] | 90 |
        """
    )

    assert result.matched is True
    assert result.template_id == "vrp_hard_time_windows_milp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 175.37, abs_tol=0.01)


def test_template_solver_handles_two_task_productivity_min_cost():
    result = solve_with_template(
        """
        A construction manager allocates resources between tasks $X$ and $Y$.
        The total hours available cannot exceed 20. Each task of type $X$
        requires 5 hours and each task of type $Y$ requires 3 hours. The
        productivity score is twice the number of $X$ tasks plus four times
        the number of $Y$ tasks, which should be at least 10 points. The cost
        per unit for task $X$ is $50, and for task $Y$, it's $80. Both X and Y
        are integers. Minimize total cost.
        """
    )

    assert result.matched is True
    assert result.template_id == "two_task_productivity_min_cost_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 210)


def test_template_solver_handles_retail_department_strong_coverage():
    result = solve_with_template(
        """
        A retail manager allocates budget among four departments: $x1$, $x2$,
        $x3$, and $x4$. The total allocation for departments $x1$ and $x2$
        must be exactly 400 units, with department $x1$ not receiving more
        than 300 units and department $x2$ not receiving more than 200 units.
        The total allocation for departments $x3$ and $x4$ cannot exceed 1000
        units, with each department having a maximum limit of 500 and 700
        units respectively. The combined allocations for departments x3 and x4
        should be at least equal to those of both departments x1 and x2. For
        department x1 it's $20, for x2 it's $15, for x3 it's $10, while for x4
        it's only $5. Allocations are whole numbers. Minimize cost.
        """
    )

    assert result.matched is True
    assert result.template_id == "retail_department_strong_coverage_min_cost_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 13000)
    assert result.artifact and result.artifact["interpretation"] == "x3 >= x1+x2 and x4 >= x1+x2"


def test_template_solver_handles_three_investment_balance_min_cost():
    result = solve_with_template(
        """
        In a financial investment scenario, an investor needs to allocate
        funds across three investments: x1, x2, and x3. The total investment
        across all three cannot exceed $10000. Each unit of investment in
        options x1, x2, and x3 incurs costs of $500, $400, and $300
        respectively. Half of the funds invested in option x1 minus a quarter
        of the funds invested in option x2 should be at least $2000. The
        difference between the funds allocated to option x3 and those allocated
        to option x1 should not exceed $500. Whole-number allocations are
        required. Minimize total cost.
        """
    )

    assert result.matched is True
    assert result.template_id == "three_investment_balance_min_cost_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 2000000)


def test_template_solver_handles_paste_and_medicine_templates():
    paste = solve_with_template(
        """
        There are two specialized containers, a small and large one, used to
        make a pharmaceutical paste. The small container requires 10 units of
        water and 15 units of the powdered pill to make 20 units of the paste.
        The large container requires 20 units of water and 20 units of the
        powdered pill to make 30 units of the paste. The pharmacy has
        available 500 units of water and 700 units of the powdered pill.
        Maximize the amount of paste that can be made.
        """
    )
    medicine = solve_with_template(
        """
        My grandma must take at least 5 grams of Z1 and 10 grams of D3. One
        pill of Zodiac contains 1.3 grams of Z1 while one pill of Sunny
        contains 1.2 grams of Z1. One pill of Zodiac contains 1.5 grams of D3
        and one pill of Sunny contains 5 grams of D3. The cost per pill of
        Zodiac is $1 and the cost per pill of Sunny is $3. Fulfill the
        medicine requirement at the lowest cost.
        """
    )

    assert paste.matched is True
    assert paste.template_id == "two_container_paste_max_lp"
    assert paste.status == "optimal"
    assert math.isclose(paste.objective_value or 0, 950)
    assert medicine.matched is True
    assert medicine.template_id == "two_medicine_pill_min_cost_ilp"
    assert medicine.status == "optimal"
    assert math.isclose(medicine.objective_value or 0, 7)


def test_template_solver_handles_sand_container_max_delivery():
    result = solve_with_template(
        """
        A sand company delivers sand for playgrounds in small and large
        containers. A small container requires 1 person to unload and can hold
        20 units of sand. A large container requires 3 people to unload and can
        hold 50 units of sand. The number of small containers used must be
        thrice the number of large containers used. There must be at least 5
        small containers and 3 large containers used. The company has 100
        people available. Maximize sand delivered.
        """
    )

    assert result.matched is True
    assert result.template_id == "sand_container_max_delivery_ilp"
    assert result.status == "optimal"
    assert math.isclose(result.objective_value or 0, 1970)


def test_template_solver_handles_phase62_mamo_min_cost_models():
    crop = solve_with_template(
        """
        A farmer grows Corn, Wheat, and Soybean. Each acre of Corn, Wheat, and
        Soybean yields a profit of $3$, $4$, and $5$ respectively. The
        combined acreage for Corn (multiplied by 2) and Wheat (multiplied by
        3) cannot exceed 15000 acres. The total acreage across all three crops
        must be at least 8000 acres. The difference in acreage between Wheat
        and Soybean should not exceed 2000 acres. Acreages are integers. What
        is the minimum total profit?
        """
    )
    project = solve_with_template(
        """
        A construction manager is planning the allocation of resources across
        three different projects: X, Y, and Z. The total resources allocated
        to all three projects combined cannot exceed 100 units. The combined
        resource allocation for twice Project X and Project Y must be at
        least 30 units to ensure their viability; the sum of resources
        allocated to Projects X and Z must be at least 40 units. The cost is
        $1000 for Project X, $2000 for Project Y, and $3000 for Project Z.
        Bounds are between 0 (inclusive) and 50 (inclusive) units for Project
        X; between 0 (inclusive) and 60 (inclusive) units for Project Y;
        between 0 (inclusive) and 70 (inclusive) units for Project Z. All
        allocations are whole numbers.
        """
    )
    energy = solve_with_template(
        """
        A green energy company invests in solar and wind energy projects. The
        cost per unit of capacity for solar and wind energy projects are $50
        and $60 respectively. The combined capacity from both solar and wind
        energy projects must be at least 1000 units. The capacity from three
        times the solar energy projects minus that from wind energy cannot
        exceed 500 units. Capacities are whole numbers. What is the minimum
        total cost?
        """
    )
    facility = solve_with_template(
        """
        In a supply chain management scenario, a company allocates resources
        between facilities $X$ and $Y$. The total allocation for both
        facilities combined cannot exceed 1000 units. Three times the
        allocation for facility $X$ minus twice that of facility $Y$ should be
        at least zero. The difference in allocation between facility $X$ and
        facility $Y$ should be at least 200 units. Each unit allocated to
        facilities $X$ and $Y$ incurs costs of 50 and 30 units respectively.
        The maximum amount of resources that can be allocated to facility X is
        700 units and to facility Y is 500 units. Minimize the total cost.
        """
    )

    assert crop.template_id == "crop_diversity_profit_min_ilp"
    assert math.isclose(crop.objective_value or 0, 25000)
    assert project.template_id == "project_resource_viability_min_cost_ilp"
    assert math.isclose(project.objective_value or 0, 40000)
    assert energy.template_id == "two_energy_capacity_min_cost_ilp"
    assert math.isclose(energy.objective_value or 0, 56250)
    assert facility.template_id == "facility_resource_balance_min_cost_ilp"
    assert math.isclose(facility.objective_value or 0, 10000)


def test_template_solver_handles_phase62_nl4opt_resource_models():
    ramen = solve_with_template(
        """
        A travelling salesman only eats ramen and fries. Each pack of ramen
        contains 400 calories, 20 grams of protein, and 100 mg of sodium.
        Each pack of fries contains 300 calories, 10 grams of protein, and 75
        mg of sodium. At most 30% of his meals can be ramen. He needs at
        least 3000 calories and 80 grams of protein. Minimize sodium.
        """
    )
    fertilizer = solve_with_template(
        """
        A farmer wants to manufacture plant nutrition using fertilizers A and
        B. Each kg of fertilizer A contains 13 units of nitrogen, 5 units of
        phosphoric acid, 6 units of vitamin A and 5 units of vitamin D. Each
        kg of fertilizer B contains 8 units of nitrogen, 14 units of
        phosphoric acid, 6 units of vitamin A and 9 units of vitamin D. The
        nutrition requires a minimum 220 units of nitrogen, a minimum of 160
        units of phosphoric acid, and no more than 350 units of vitamin A.
        Minimize the amount of vitamin D.
        """
    )
    mail = solve_with_template(
        """
        A village delivers mail by runners or canoers. Runners can carry three
        bags of mail each time and takes 4 hours. Canoers can carry ten bags
        of mail each time and takes 2 hours. At most 33% of deliveries can be
        by canoe. The village can spare at most 200 total hours and at least
        4 runners must be used. Maximize the total amount of mail delivered.
        """
    )
    ice_cream = solve_with_template(
        """
        An ice cream store makes chocolate and vanilla ice cream by the gallon.
        In a week, they must make at least 5 gallons of each type but at most
        10 gallons of chocolate ice cream and at most 8 gallons of vanilla ice
        cream. It takes 1 hour to produce a gallon of chocolate ice cream and
        2 hours to produce a gallon of vanilla ice cream. In a week, 30 hours
        are available. At least 6 workers are needed. The profit per gallon of
        chocolate ice cream is $200 and the profit per gallon of vanilla ice
        cream is $300. How many gallons should be made to maximize profit?
        """
    )

    assert ramen.template_id == "two_food_sodium_min_lp"
    assert math.isclose(ramen.objective_value or 0, 750)
    assert fertilizer.template_id == "two_fertilizer_vitamin_min_lp"
    assert math.isclose(fertilizer.objective_value or 0, 125.49295774647887)
    assert mail.template_id == "runner_canoe_mail_max_ilp"
    assert math.isclose(mail.objective_value or 0, 310)
    assert ice_cream.template_id == "ice_cream_profit_bounds_lp"
    assert math.isclose(ice_cream.objective_value or 0, 4400)


def test_template_solver_rejects_phase62_conflicting_gold_cases():
    ship = solve_with_template(
        """
        A shipping company allocates resources across four ship types x1, x2,
        x3, and x4. The total number of x1 and x2 ships cannot exceed 1000,
        and x3 and x4 is limited to 1500. The number of x1 ships must be at
        least twice half the number of x3 ships, and no more than half the
        number of x4 ships can be added without exceeding x2 by 400. Costs
        are positive and the objective is to minimize total operating cost.
        """
    )
    pills = solve_with_template(
        """
        Sleep inducing medicine and anti-inflammatory medicine is found in pill
        A and pill B. One pill A contains 3 units of sleep inducing medicine
        and 5 units of anti-inflammatory medicine. One pill B contains 6 units
        of sleep-inducing medicine and 1 unit of anti-inflammatory medicine.
        The cost per pill for pill A is $4 and pill B is $5. A patient must
        get at least 40 units of sleep-inducing medicine and 50 units of
        anti-inflammatory medicine. Formulate an LP to minimize cost.
        """
    )

    assert ship.matched is False
    assert pills.matched is False


def test_template_solver_handles_phase63_industryor_models():
    furnace = solve_with_template(
        """
        Two steel furnaces each use two methods of steelmaking simultaneously.
        The first method takes $a=2$ hours per furnace and costs $m=50$ in
        fuel expenses; the second method takes $b=3$ hours per furnace and
        costs $n=70$ in fuel expenses. Each furnace produces $k=10$ tons of
        steel regardless of the method used, and at least $d=30$ tons must be
        produced within $c=12$ hours. Minimize fuel expenses.
        """
    )
    mall = solve_with_template(
        """
        Changjiang Comprehensive Shopping Mall has 5000 m² of space for lease
        and wants to maximize total rental income. Each store pays 20% of its
        annual profit as rent to the mall.

        | Code | Store Type | Area per Shop / m² | Min | Max | 1 Store | 2 Stores | 3 Stores |
        |------|------------|--------------------|-----|-----|---------|----------|----------|
        | 1    | Jewelry    | 250                | 1   | 3   | 9       | 8        | 7        |
        | 2    | Shoes & Hats | 350              | 1   | 2   | 10      | 9        | -        |
        | 3    | General Merchandise | 800      | 1   | 3   | 27      | 21       | 20       |
        | 4    | Bookstore  | 400                | 0   | 2   | 16      | 10       | -        |
        | 5    | Catering   | 500                | 1   | 3   | 17      | 15       | 12       |
        """
    )
    fruit = solve_with_template(
        """
        Vicky and David have just bought a farm in the Yarra Valley, and they
        are considering using it to grow apples, pears, oranges, and lemons.
        The profit for growing one acre of apples is $2000, for one acre of
        pears is $1800, for one acre of oranges is $2200, and for one acre of
        lemons is $3000. The farm has a total area of 120 acres. The land used
        to grow apples should be at least twice the land used to grow pears.
        The land used to grow apples should be at least three times the land
        used to grow lemons. The land used to grow oranges must be twice the
        land used to grow lemons. Vicky and David are unwilling to grow more
        than two types of fruit.
        """
    )

    assert furnace.template_id == "steel_furnace_method_min_cost_lp"
    assert math.isclose(furnace.objective_value or 0, 150)
    assert mall.template_id == "mall_store_lease_piecewise_enum"
    assert math.isclose(mall.objective_value or 0, 28, rel_tol=0.05)
    assert fruit.template_id == "fruit_farm_two_type_profit_lp_enum"
    assert math.isclose(fruit.objective_value or 0, 240000)


def test_template_solver_handles_phase63_mamo_models():
    staffing = solve_with_template(
        """
        In a human resources planning scenario, a company allocates employees
        across three departments: $X1$, $X2$, and $X3$. The total number of
        employees that can be allocated is limited to 10. Department $X1$
        requires a minimum of 2 employees, department $X2$ needs at least 3
        employees, and department $X3$ requires no fewer than 5 employees.
        Each employee in departments $X1$, $X2$, and $X3$ costs the company
        $50000, $60000, and $70000 respectively. Department X1 can have
        between 0 and 6 employees; Department X2 can have between 0 and 4;
        Department X3 can have between 0 and 5. Minimize the total cost.
        """
    )
    diet = solve_with_template(
        """
        Daily requirements are 78 grams of protein, 140 grams of carbs, and
        1537 calories. Choose food servings at minimum cost.

        - Chicken Breast: For $4, you'll get 14 grams of protein, 4 grams of carbohydrates, and 275 calories.
        - Brown Rice: A $10 serving gives you 11 grams of protein, 17 grams of carbohydrates, and 151 calories.
        - Tofu: For $6, you can get 20 grams of protein, 12 carbs, and 155 calories.
        - Spinach: At only $1, you get 6 grams of protein, 20 carbs, and 106 calories.
        - Almonds: For $8, you get 9 grams of protein, 11 grams of carbohydrates, and 279 calories.
        - Salmon: Finally, for $6, you get 20 grams of protein, 19 grams of carbohydrates, and 93 calories.

        What is the least amount of money needed?
        """
    )
    salary = solve_with_template(
        """
        A human resources manager is planning to allocate employees across four
        different departments: x1, x2, x3, and x4. The objective is to minimize
        the total salary cost associated with these departments. Each employee
        in departments x1, x2, x3 and x4 earns a salary of $5000, $4000, $6000
        and $7000 per month respectively. The combined number of employees in
        departments x1 and x2 cannot exceed 100. The combined number of
        employees in departments x3 and x4 cannot exceed 80. The number of
        employees in department X1 must be at least 20 more than half the
        number of employees in department X3. The difference between the number
        of employees in department X4 and X2 should not exceed 30. x1 can have
        up to 50 employees, x2 can have up to 60, x3 can have up to 40, x4 can
        also have up to 40. Counts must be whole numbers.
        """
    )
    training = solve_with_template(
        """
        In a sports team, allocate hours for four training areas: x1 strength
        and conditioning, x2 skill development, x3 strategy learning, and x4
        recovery sessions. The cost per hour of each area are $20$, $30$, $50$
        and $60 respectively. Only whole number allocations are feasible. The
        combined hours of strength and conditioning (x1) and skill development
        (x2) must be at least 5. The difference between strategy learning (x3)
        and recovery sessions (x4) cannot exceed 10 hours. Twice the hours of
        strength and conditioning plus three times skill development minus
        recovery session should be at least 8. The difference between strength
        and conditioning (x1) hours and strategy learning(x3) should not
        exceed 6. Strength & Conditioning: [0,15] Skill Development: [0,12]
        Strategy Learning: [0,10] Recovery Sessions: [0,8]. Find the minimum
        total cost.
        """
    )

    assert staffing.template_id == "three_department_staffing_min_cost_ilp"
    assert math.isclose(staffing.objective_value or 0, 630000)
    assert diet.template_id == "multifood_integer_diet_min_cost_ilp"
    assert math.isclose(diet.objective_value or 0, 15)
    assert salary.template_id == "four_department_salary_balance_min_cost_ilp"
    assert math.isclose(salary.objective_value or 0, 100000)
    assert training.template_id == "four_training_area_min_cost_ilp"
    assert math.isclose(training.objective_value or 0, 100)


def test_template_solver_handles_phase63_nl4opt_models():
    sandwich = solve_with_template(
        """
        A breakfast joint makes regular and special sandwiches. Each regular
        sandwich requires 2 eggs and 3 slices of bacon. Each special sandwich
        requires 3 eggs and 5 slices of bacon. The joint has a total of 40 eggs
        and 70 slices of bacon. It makes a profit of $3 per regular sandwich
        and a profit of $4 per special sandwich. Maximize profit.
        """
    )
    vans = solve_with_template(
        """
        A shipping company can purchase regular and hybrid vans. A regular van
        can deliver 500 packages per day and produces 200 units of pollutants.
        A hybrid van can deliver 300 packages per day and produces 100 units of
        pollutants. They can produce at most 7000 units of pollutants per day
        and need to deliver at least 20000 packages per day. Minimize the total
        number of vans needed.
        """
    )
    medication = solve_with_template(
        """
        A patient takes anxiety medication and anti-depressants. Each unit of
        anxiety medication takes 3 minutes to be effective while each unit of
        anti-depressant takes 5 minutes. The patient must take at least 100
        units of medication and at least 30 should be anxiety medication. The
        patient can take at most twice the amount of anxiety medication as
        anti-depressants. Minimize the total time.
        """
    )
    snow = solve_with_template(
        """
        A city employs seasonal and permanent snow removers. A seasonal snow
        remover works 6 hours per shift and gets paid $120. A permanent snow
        remover works 10 hours per shift and gets paid $250. The city needs
        300 hours of snow remover labor and has a budget of $6500. Minimize
        the total number of snow removers.
        """
    )

    assert sandwich.template_id == "two_sandwich_profit_max_lp"
    assert math.isclose(sandwich.objective_value or 0, 60)
    assert vans.template_id == "two_van_min_count_ilp"
    assert math.isclose(vans.objective_value or 0, 60)
    assert medication.template_id == "two_medication_time_min_ilp"
    assert math.isclose(medication.objective_value or 0, 368)
    assert snow.template_id == "snow_remover_min_count_ilp"
    assert math.isclose(snow.objective_value or 0, 44)


def test_template_solver_handles_phase64_course_and_mamo_models():
    courses = solve_with_template(
        """
        A master's student is required to select two courses in mathematics,
        two in operations research, and two in computer science from Calculus,
        Operations Research, Data Structures, Management Statistics, Computer
        Simulation, Computer Programming, and Forecasting. Some courses have
        prerequisites: Computer Simulation or Data Structures requires Computer
        Programming first, Management Statistics requires Calculus first, and
        Forecasting requires Management Statistics first. What is the minimum
        number of courses a master's student must take?
        """
    )
    marketing = solve_with_template(
        """
        A marketing firm distributes budget between advertising channels X, Y,
        and Z. Channel X and Y combined cannot exceed $1000. Channels Y and Z
        cannot surpass $800. Channels X and Z must yield a minimum expenditure
        of at least $500. The cost per unit of effectiveness for channel X is
        $5, for channel Y is $4, and for channel Z is $3. Minimize total cost.
        """
    )
    construction = solve_with_template(
        """
        A construction company allocates resources across three different
        projects: $X, Y$, and $Z$. The total resource allocation across all
        three projects cannot exceed 1000 units. Project $X$ requires a
        minimum allocation of 200 units, project $Y$ needs at least 150 units,
        and Project $Z$ requires a minimum allocation of 100 units. Project X
        should not exceed that for project Y by more than 50 units. Project Y
        should be at least 20 units more than that for project Z. The costs
        are $300$ for project X, $500$ for project Y, and $200$ for project Z.
        Minimize the total cost.
        """
    )
    telecom = solve_with_template(
        """
        A telecommunications company allocates resources between two projects:
        X and Y. Each unit of project X costs 4 units, while each unit allocated
        to project Y costs 5 units. The total resource allocation across both
        projects has a maximum limit of 100 units, with three times the
        allocation for project Y included in this calculation. The combined
        allocation of twice that for project X and that for project Y must be
        at least 50 units. The difference between allocations for project X
        and Y should not exceed 20 units. Project X can't exceed 60 units while
        Project Y can't exceed 30 units. Minimize total cost.
        """
    )
    tourism = solve_with_template(
        """
        A tourism agency is planning to distribute the annual budget among
        three projects: X, Y, and Z. The total budget across all three projects
        cannot exceed 300 units. Project X requires a minimum investment of 80
        units, while project Y needs at least 60 units. Project Z demands a
        minimum of 40 units. Each unit of investment in projects X, Y, and Z
        incurs different costs estimated as 50, 70, and 100 units respectively.
        The agency aims to minimize the total cost.
        """
    )

    assert courses.template_id == "course_prerequisite_cover_min_enum"
    assert math.isclose(courses.objective_value or 0, 4)
    assert marketing.template_id == "three_channel_effort_min_cost_ilp"
    assert math.isclose(marketing.objective_value or 0, 1500)
    assert construction.template_id == "three_project_minimum_allocation_min_cost_ilp"
    assert math.isclose(construction.objective_value or 0, 155000)
    assert telecom.template_id == "two_project_weighted_resource_min_cost_ilp"
    assert math.isclose(telecom.objective_value or 0, 112)
    assert tourism.template_id == "three_project_lower_bound_budget_min_cost_ilp"
    assert math.isclose(tourism.objective_value or 0, 12200)


def test_template_solver_handles_phase64_nl4opt_models():
    stores = solve_with_template(
        """
        A clothing company can sell in a retail store or a factory outlet. A
        retail store brings in 200 customers every day and requires 6 employees
        to operate. A factory outlet brings in 80 customers every day and
        requires 4 employees to run. There must be at least 1200 customers
        every day, and executives can make available 50 employees. Reduce the
        number of stores that must be open.
        """
    )
    tea = solve_with_template(
        """
        A tea estate has available 500 acres of land and can pick tea leaves
        using a traditional machine or modern machine. The traditional machine
        can pick 30 kg of tea leaves, creates 10 kg of waste, and requires 20
        liters of fuel per acre. The modern machine can pick 40 kg, creates 15
        kg of waste, and requires 15 liters of fuel. The estate has available
        9000 liters of fuel and can handle at most 6000 kg of waste. Maximize
        the amount of tea leaves that can be picked.
        """
    )
    experiments = solve_with_template(
        """
        A chemistry teacher teaches experiment 1 and experiment 2. In experiment
        1, 3 units of red liquid and 4 units of blue liquid mix to create 5
        units of green gas. In experiment 2, 5 units of red liquid and 3 units
        of blue liquid mix to create 6 units of the green gas. Experiment 1
        produces 1 units of smelly gas while experiment 2 produces 2 units. The
        lab has available 80 units of red liquid and 70 units of blue liquid. At
        most 10 units of smelly gas can be produced. Maximize the total amount
        of green gas produced.
        """
    )
    transport = solve_with_template(
        """
        An exporter uses ships and planes to transport goods. A ship can take
        40 containers worth of goods and uses 500 liters of fuel per trip. A
        plane can take 20 containers worth of goods and uses 300 liters of fuel
        per trip. The company needs to transport at least 500 containers worth
        of goods. At most 10 plane trips can be made and a minimum of 50% of
        the trips made must be by ship. Minimize the total amount of fuel.
        """
    )

    assert stores.template_id == "two_store_customer_min_count_ilp"
    assert math.isclose(stores.objective_value or 0, 6)
    assert tea.template_id == "two_machine_tea_leaf_max_lp"
    assert math.isclose(tea.objective_value or 0, 17000)
    assert experiments.template_id == "two_experiment_green_gas_max_ilp"
    assert math.isclose(experiments.objective_value or 0, 50)
    assert transport.template_id == "ship_plane_fuel_min_ilp"
    assert math.isclose(transport.objective_value or 0, 6300)


def test_template_solver_handles_phase65_industry_and_mamo_models():
    meal = solve_with_template(
        """
        Mary is planning her dinner tonight. Every 100 grams of okra contains
        3.2 grams of fiber, every 100 grams of carrots contains 2.7 grams of
        fiber, every 100 grams of celery contains 1.6 grams of fiber, and every
        100 grams of cabbage contains 2 grams of fiber. How many grams of each
        type of food should Mary buy to maximize her fiber intake? She is
        considering choosing one among salmon, beef, and pork as a protein
        source. She also considers choosing at least two kinds of vegetables
        among okra, carrots, celery, and cabbage. The price of salmon is $4 per
        100 grams, beef is $3.6 per 100 grams, pork is $1.8 per 100 grams. The
        price of okra is $2.6 per 100 grams, carrots are $1.2 per 100 grams,
        celery is $1.6 per 100 grams, and cabbage is $2.3 per 100 grams. Mary
        has a budget of $15 for this meal. The total food intake should be 600
        grams.
        """
    )
    routes = solve_with_template(
        """
        A shipping company needs to allocate their resources among three
        routes: $X, Y$, and $Z$. The total resource allocation across all three
        routes cannot exceed 100 units. To ensure a balanced service coverage,
        the allocation for route X must be at least 10 units more than that for
        route Y. Additionally, the combined resource allocation for routes Y
        and Z cannot exceed 70 units. Each unit of resource allocated to routes
        $X, Y$, and $Z$ incurs costs of 5, 3, and 4 units respectively. Minimize
        the total cost while keeping whole-number allocations.
        """
    )
    energy = solve_with_template(
        """
        An energy company is planning to allocate resources across three
        projects: $x1, x2$, and $x3$. The total investment across all three
        projects cannot exceed 1000 units. The difference between twice the
        resource allocated for project $x1$ and thrice that of project $x2$
        should be at least 200 units. The resource allocated for project $x1$
        cannot exceed by more than 300 units, half of that allocated for project
        $x3$. The combined resource allocation for project $x2$, two and a half
        times as much minus the allocations for both projects $x1$ and $x3$,
        should not fall below -500 units. Each unit of investment in projects
        $x1, x2$, and $x3$ yields different returns or costs, quantified as 50,
        60, and 70 units respectively. Find the minimum total investment cost.
        """
    )

    assert meal.template_id == "meal_fiber_selection_lp_enum"
    assert math.isclose(meal.objective_value or 0, 18.6943, rel_tol=1e-4)
    assert routes.template_id == "three_route_balance_min_cost_ilp"
    assert math.isclose(routes.objective_value or 0, 50)
    assert energy.template_id == "energy_project_linear_min_cost_ilp"
    assert math.isclose(energy.objective_value or 0, 5000)


def test_template_solver_handles_phase65_nl4opt_models():
    printers = solve_with_template(
        """
        An office supply company makes two types of printers: color printers
        and black and white printers. The color printer team can produce at
        most 20 color printers per day while the black and white printer team
        can produce at most 30 black and white printers per day. Both teams
        require use of the same paper tray installing machine and this machine
        can make at most 35 printers of either type each day. Color printers
        generate a profit of $200 per printer while black and white printers
        generate a profit of $70 per printer. Maximize profit.
        """
    )
    animals = solve_with_template(
        """
        A company delivers packages to customers on camels and horses. A camel
        can carry 50 packages while a horse can carry 60 packages. A camel
        requires 20 units of food while a horse requires 30 units of food. The
        company needs to deliver at least 1000 packages and they have 450 units
        of food available. The number of horses cannot exceed the number of
        camels. Minimize the total number of animals.
        """
    )
    packages = solve_with_template(
        """
        A grocery store wants to liquidate its stock of 10 apples, 20 bananas,
        and 80 grapes. They can propose a banana-haters package with 6 apples
        and 30 grapes and this package will bring a profit of 6 euros. They can
        prepare a combo package with 5 apples, 6 bananas, and 20 grapes,
        yielding a profit of 7 euros. Maximize net profit.
        """
    )
    stores = solve_with_template(
        """
        A sandwich company can open two types of stores, a dine-in place and a
        food-truck. A dine-in place can make 100 sandwiches per day and requires
        8 employees to operate. A food-truck can make 50 sandwiches per day and
        requires 3 employees to operate. The company must make at least 500
        sandwiches per day but they only have available 35 employees. Minimize
        the total number of stores.
        """
    )

    assert printers.template_id == "two_printer_shared_machine_profit_lp"
    assert math.isclose(printers.objective_value or 0, 5050)
    assert animals.template_id == "two_animal_package_min_count_ilp"
    assert math.isclose(animals.objective_value or 0, 19)
    assert packages.template_id == "two_package_stock_profit_lp"
    assert math.isclose(packages.objective_value or 0, 14)
    assert stores.template_id == "two_store_capacity_min_count_ilp"
    assert math.isclose(stores.objective_value or 0, 8)


def test_template_solver_handles_phase66_industry_and_mamo_models():
    meal = solve_with_template(
        """
        Mary is planning tonight's dinner. She wants to choose a combination of
        protein and vegetables to maximize her protein intake for the meal. Her
        protein options are chicken, salmon, and tofu, which can be bought in
        any quantity. Chicken: 23g protein, $3.00 cost, per 100g. Salmon: 20g
        protein, $5.00 cost, per 100g. Tofu: 8g protein, $1.50 cost, per 100g.
        She must select at least three different types of vegetables. Broccoli
        (100g pack): 2.8g protein, $1.20 cost. Carrots (100g pack): 0.9g
        protein, $0.80 cost. Spinach (100g pack): 2.9g protein, $1.50 cost.
        Bell Pepper (100g pack): 1.0g protein, $1.00 cost. Mushrooms (100g
        pack): 3.1g protein, $2.00 cost. Her total budget is $20. The total
        weight of all food must not exceed 800 grams.
        """
    )
    project = solve_with_template(
        """
        A telecommunications company is planning to allocate resources between
        Project $X$ and Project $Y$. A resource unit for project $X$ costs
        $2000 and for project $Y$, $3000. The performance score calculated as
        5 times the resources allocated to project X plus 10 times those
        allocated to project Y should be at least 50 points. The total usage
        calculated as 3 times the resources used by project X plus 4 times
        those used by project Y cannot exceed 40 units. X and Y are integers.
        """
    )
    environmental = solve_with_template(
        """
        An environmental organization is planning to allocate funds between
        project $X$ for reforestation and project $Y$ for ocean cleanup. The
        total budget for both projects combined cannot exceed $10000. The
        combined impact score, calculated as 10 times the budget for project X
        plus 4 times the budget for project Y, must be at least 40000 points.
        Each dollar spent on projects X and Y is quantified as 50 and 30 points
        per dollar respectively. Minimize the total cost with integer budgets.
        """
    )
    vehicles = solve_with_template(
        """
        A transportation company has three types of vehicles: X, Y, and Z. The
        costs are $5, $4, and $3 per vehicle for X, Y, and Z respectively. The
        combined number of all types of vehicles cannot exceed 1000. The number
        of twice type X plus type Y vehicles must be at least 200. The number
        of type X plus type Z vehicles cannot exceed 300. Minimize total cost.
        """
    )

    assert meal.template_id == "meal_protein_pack_selection_lp_enum"
    assert math.isclose(meal.objective_value or 0, 123.8)
    assert project.template_id == "two_project_performance_resource_min_cost_ilp"
    assert math.isclose(project.objective_value or 0, 15000)
    assert environmental.template_id == "environmental_project_budget_min_cost_ilp"
    assert math.isclose(environmental.objective_value or 0, 200000)
    assert vehicles.template_id == "three_vehicle_operating_min_cost_ilp"
    assert math.isclose(vehicles.objective_value or 0, 500)


def test_template_solver_handles_phase66_network_and_nl4opt_models():
    network = solve_with_template(
        """
        Compute the maximum amount that can be distributed from Node 0 to the
        Final Distribution Center. The connections and capacities are:
        - From the Power Plant (Node 0): Electricity can be sent to Node 1
          (2 kWh), Node 2 (1 kWh), Node 3 (1 kWh), Node 4 (8 kWh), Node 5
          (17 kWh), Node 6 (6 kWh), and Node 7 (7 kWh).
        - From Node 1: Electricity can flow to Node 7 (19 kWh).
        - From Node 4: Electricity can be dispatched to Node 7 (14 kWh).
        - From Node 5: Electricity can move to Node 7 (6 kWh).
        - From Node 6: Electricity can be sent to Node 7 (10 kWh).
        - From Node 7 (Final Distribution Center): Electricity can return to
          Node 0 (20 kWh).
        What is the maximum amount from the source to the final distribution?
        """
    )
    mobile = solve_with_template(
        """
        Large mobile production units can hold 6 people and takes up 2 parking
        spots whereas small mobile production units can hold only 2 people and
        takes up 1 parking spot. At least 5 units must be small mobile units.
        Large mobile production units must make up at least 75% of all
        vehicles. The movie needs to transport 80 people. Minimize the total
        number of parking spots.
        """
    )
    daycare = solve_with_template(
        """
        A daycare center can use bus or a personal car. A bus can carry 9
        children while a personal car can carry 4 children. The daycare has to
        pick up at least 100 children. There must be more buses than personal
        cars and at least 5 personal cars. Minimize total vehicles.
        """
    )
    pizza = solve_with_template(
        """
        Large pizzas require 12 units of dough, and 5 units of toppings.
        Medium pizzas require 8 units of dough, and 4 units of toppings. Large
        pizzas take 12 minutes to bake, medium pizzas require 8 minutes to
        bake. The restaurant must use at least 10000 units of dough and 4400
        units of toppings. At least 200 medium pizzas must be made. At least
        two times as many large pizzas should be made than medium pizzas.
        Reduce time spent baking.
        """
    )
    milk_tea = solve_with_template(
        """
        A milk tea shop owner would like to sell black milk tea and matcha milk
        tea. A bottle of black milk tea contains 600 grams of milk and 10 grams
        of honey, whereas a bottle of matcha milk tea contains 525 grams of
        milk and 5 grams of honey. The profit from each bottle of black milk
        tea sold is $7.5 and the profit from each bottle of matcha milk tea
        sold is $5. His available stock is 30000 grams of milk and 500 grams of
        honey. Maximize profits.
        """
    )

    assert network.template_id == "max_flow_network"
    assert math.isclose(network.objective_value or 0, 29)
    assert mobile.template_id == "mobile_unit_parking_min_ilp"
    assert math.isclose(mobile.objective_value or 0, 35)
    assert daycare.template_id == "bus_car_child_pickup_min_count_ilp"
    assert math.isclose(daycare.objective_value or 0, 14)
    assert pizza.template_id == "pizza_baking_time_min_ilp"
    assert math.isclose(pizza.objective_value or 0, 10060)
    assert milk_tea.template_id == "two_milk_tea_profit_lp"
    assert math.isclose(milk_tea.objective_value or 0, 375)


def test_template_solver_handles_phase67_industry_and_mamo_models():
    farm = solve_with_template(
        """
        Tom and Jerry bought a farm in Sunshine Valley and can plant corn,
        wheat, soybeans, and sorghum. The profit per acre for planting corn is
        $1500, the profit per acre for planting wheat is $1200, the profit per
        acre for planting soybeans is $1800, and the profit per acre for
        planting sorghum is $1600. The farm has a total area of 100 acres. The
        land area used for planting corn must be at least twice the land area
        used for planting wheat. The land area used for planting soybeans must
        be at least half the land area used for planting sorghum. The land area
        used for planting wheat must be three times the land area used for
        planting sorghum. Maximize profit.
        """
    )
    byproduct = solve_with_template(
        r"""
        There are $\mathrm{A}$ and $\mathrm{B}$ two products, both requiring
        two successive chemical reaction processes. Each unit of product
        $\mathrm{A}$ needs 2 hours for the first process and 3 hours for the
        second process. Each unit of product $\mathrm{B}$ needs 3 hours for the
        first process and 4 hours for the second process. Available time for
        the first process is 16 hours, and available time for the second
        process is 24 hours. For each unit of product $\mathrm{B}$ produced, 2
        units of by-product $\mathrm{C}$ are generated. By-product
        $\mathrm{C}$ can be sold up to 5 units, and the rest must be disposed
        of at a cost of 2 yuan per unit. Each unit of product $\mathrm{A}$ sold
        yields a profit of 4 yuan, each unit of product $\mathrm{B}$ yields a
        profit of 10 yuan, and each unit of by-product $\mathrm{C}$ sold yields
        a profit of 3 yuan. Maximize total profit.
        """
    )
    workforce = solve_with_template(
        """
        A company requires skilled workers and laborers for three tasks. The
        company must choose exactly one method to complete each task. Task 1
        (Requires 8,400 effective hours): Method A is skilled workers alone;
        Method B is groups of one skilled worker and two laborers with a fixed
        weekly setup cost of 500 yuan. Task 2 (Requires 10,800 effective hours):
        Method A is skilled workers alone; Method B is laborers alone. Task 3
        (Requires 18,000 effective hours): Method A is groups of five laborers;
        Method B is groups of one skilled worker and three laborers. Weekly
        wages: 100 yuan for skilled workers, 80 yuan for laborers. Effective
        working hours per week: 42 hours for skilled workers, 36 hours for
        laborers. The number of workers is limited to a maximum of 400 skilled
        workers and 800 laborers. Exclusion Rule: if Task 1 uses Method B, then
        Task 3 cannot use Method A. Minimum Assignment: if Method B is chosen
        for Task 3, a minimum of 20 skilled workers must be assigned to it.
        Hiring Policy: skilled workers hired cannot exceed 60% of the total
        number of laborers hired. Minimize total weekly cost.
        """
    )
    warehouse = solve_with_template(
        """
        A shipping company has three warehouses, $X$, $Y$ and $Z$. The cost per
        resource being $5 for warehouse $X$, $10 for warehouse $Y$, and $7 for
        warehouse $Z$. The total number of resources available is limited to
        1000. Warehouse $X$ requires at least 200 resources. Warehouse $Y$ can
        handle no more than 400 resources. Warehouse $Z$ requires at least 150
        resources. Minimize total cost with whole-number resources.
        """
    )
    task_time = solve_with_template(
        """
        A manager is scheduling three types of tasks: $X, Y$, and $Z$. Each task
        type requires a different amount of time to complete, with task $X$
        taking 1 hour, task $Y$ taking 2 hours, and task $Z$ taking 3 hours.
        The combined time for twice as many task $X$ plus task $Y$ cannot
        exceed 6 hours. The combined time for task $X$ and task $Z$ must not
        exceed 5 hours. The total time spent on tasks $Y$ and $Z$ must be at
        least 7 hours. Minimize total number of hours.
        """
    )

    assert farm.template_id == "farm_four_crop_ratio_profit_lp"
    assert math.isclose(farm.objective_value or 0, 180000)
    assert byproduct.template_id == "chemical_byproduct_profit_lp"
    assert math.isclose(byproduct.objective_value or 0, 57)
    assert workforce.template_id == "three_task_method_selection_min_cost_enum"
    assert math.isclose(workforce.objective_value or 0, 84000)
    assert warehouse.template_id == "warehouse_resource_lower_bound_min_cost_ilp"
    assert math.isclose(warehouse.objective_value or 0, 2050)
    assert task_time.template_id == "three_task_time_min_hours_ilp"
    assert math.isclose(task_time.objective_value or 0, 15)


def test_template_solver_handles_phase67_nl4opt_models():
    souvenirs = solve_with_template(
        """
        A souvenir shop makes wooden elephants and tigers with plastic
        ornaments. Each elephant requires 50 grams of wood and 20 grams of
        plastic. Each tiger requires 40 grams of wood and 30 grams of plastic.
        In a week, 5000 grams of wood and 4000 grams of plastic are available.
        The profit per elephant sold is $5 and the profit per tiger sold is $4.
        Maximize profit.
        """
    )
    transport = solve_with_template(
        """
        A shipping company need to transport packages by either truck or car. A
        truck can transport 50 packages per trip and uses 20 liters of gas per
        trip. A car can transport 30 packages per trip and uses 15 liters of gas
        per trip. There can be at most 5 truck trips made and at least 30% of
        all the trips must be made by car. The company needs to transport at
        least 500 packages. Minimize gas consumed.
        """
    )
    rice = solve_with_template(
        """
        Grain is transported in either large bags or tiny bags. Large bags can
        hold 25 kg of grain and requires 4 units of energy to transport. Tiny
        bags can hold 6 kg of grain and requires 1.5 units of energy to
        transport. The process has access to 110 units of energy. There must be
        twice as many large bags as tiny bags of rice. Additionally, there must
        be at least 20 tiny bags of rice. Maximize the total amount of grain in
        weight.
        """
    )

    assert souvenirs.template_id == "souvenir_elephant_tiger_profit_lp"
    assert math.isclose(souvenirs.objective_value or 0, 500)
    assert transport.template_id == "truck_car_gas_min_ilp"
    assert math.isclose(transport.objective_value or 0, 230)
    assert rice.template_id == "rice_bag_weight_max_ilp"
    assert rice.status == "infeasible"


def test_template_solver_handles_phase68_industry_models():
    toys = solve_with_template(
        """
        Bright Future Toys wants to build and sell robots, model cars, building
        blocks, and dolls. The profit for each robot sold is $15, for each
        model car sold is $8, for each set of building blocks sold is $12, and
        for each doll sold is $5. There are 1200 units of plastic available.
        Each robot requires 30 units of plastic, each model car requires 10
        units of plastic, each set of building blocks requires 20 units of
        plastic, and each doll requires 15 units of plastic. There are 800
        units of electronic components available. Each robot requires 8 units
        of electronic components, each model car requires 5 units of
        electronic components, each set of building blocks requires 3 units of
        electronic components, and each doll requires 2 units of electronic
        components. If Bright Future Toys manufactures robots, they will not
        manufacture dolls. If they manufacture model cars, they will also
        manufacture building blocks. The number of dolls manufactured cannot
        exceed the number of model cars manufactured. Maximize profit.
        """
    )
    fixed_cost = solve_with_template(
        """
        Hongdou Clothing Factory produces shirts, short-sleeved shirts, and
        casual clothes.

        | Product Name | Labor per unit | Material per unit | Selling Price | Variable Cost |
        |--------------|----------------|-------------------|---------------|---------------|
        | Shirt        | 3              | 4                 | 120           | 60            |
        | Short-sleeve | 2              | 3                 | 80            | 40            |
        | Casual Cloth | 6              | 6                 | 180           | 80            |

        Available labor is 1500 units, available material is 1600 units, and
        weekly fixed costs are 2000, 1500, and 1000 respectively. Maximize
        profit.
        """
    )
    equipment = solve_with_template(
        """
        A factory produces Product I, Product II, and Product III. Each product
        must undergo stage A and stage B on compatible equipment. Maximize
        total profit.

        | Equipment | Product I | Product II | Product III | Effective Machine Hours | Processing Cost per Machine Hour (Yuan/hour) |
        | :--- | :--- | :--- | :--- | :--- | :--- |
        | A1 | 5 | 10 | - | 6000 | 0.05 |
        | A2 | 7 | 9 | 12 | 10000 | 0.03 |
        | B1 | 6 | 8 | - | 4000 | 0.06 |
        | B2 | 4 | - | 11 | 7000 | 0.11 |
        | B3 | 7 | - | - | 4000 | 0.05 |
        | Raw Material Cost (Yuan/piece) | 0.25 | 0.35 | 0.5 | - | - |
        | Unit Price (Yuan/piece) | 1.25 | 2 | 2.8 | - | - |
        """
    )

    assert toys.template_id == "toy_product_logic_profit_milp"
    assert math.isclose(toys.objective_value or 0, 956)
    assert fixed_cost.template_id == "product_table_fixed_cost_profit_lp"
    assert math.isclose(fixed_cost.objective_value or 0, 21500)
    assert equipment.template_id == "multi_machine_process_profit_lp"
    assert math.isclose(equipment.objective_value or 0, 1190.5665024630543)


def test_template_solver_handles_phase68_mamo_models():
    stocks = solve_with_template(
        """
        A financial advisor allocates investment funds between stocks and
        bonds. The combined amount invested in twice the number of stocks and
        bonds should be at least $50000. The difference between the amount
        invested in stocks and twice that of bonds cannot exceed $20000. The
        cost per unit for stocks is $4 while it is $3 for bonds. Minimize the
        total cost, using whole numbers.
        """
    )
    services = solve_with_template(
        """
        A transportation company allocates resources between service X and
        service Y. The total resources available for both services combined
        cannot exceed 20 units. The sum of the resources allocated to service X
        and twice that for service Y must be at least 10 units. The difference
        in resources between twice that of service X and service Y should not
        exceed 15 units. The cost for service X is 5 units and for service Y is
        4 units. Minimize the total cost, and allocations are integers.
        """
    )
    projects = solve_with_template(
        """
        An environmental agency allocates resources to project X, project Y,
        and project Z. Costs being $50 for project X, $60 for project Y and
        $70 for project Z. The combined resource allocation for twice the
        allocation for X plus thrice the allocation for Y and Z cannot be less
        than 500 units. The sum of allocations for X, Y and four times the
        allocation for Z should not exceed 800 units. The difference between
        allocations of X and Y added with that of Z should be at least 100
        units. 0 <= x <= 200, 0 <= y <= 150, 0 <= z <= 300. Minimize the total
        cost in whole numbers.
        """
    )
    properties = solve_with_template(
        """
        A real estate developer allocates investments across residential (x),
        commercial (y), industrial (z), and mixed-use (w) properties. Costs
        being $30000, $40000, $50000, and $60000 for x, y, z, and w
        respectively. The combined number of residential and commercial
        properties cannot exceed 20. The combined number of industrial and
        mixed-use properties cannot exceed 30. At least 10 properties must be
        either residential or industrial. At least 15 properties must be either
        commercial or mixed-use. Residential (x) : [0 , 15], Commercial (y) :
        [0 , 20], Industrial (z) : [0 , 25], Mixed Use (w) : [0 , 30].
        Minimize the total cost with whole numbers.
        """
    )
    tsp = solve_with_template(
        """
        The driver must deliver to each shop and then return to the starting
        point. The driver can visit each shop only once. The goal is to
        minimize the total travel cost. The cost to travel from Shop1 to Shop2
        is 13 units, to Shop3 is 87 units, to Shop4 is 76 units, and to Shop5
        is 50 units. From Shop2, it costs 13 units to reach Shop1, 15 units to
        get to Shop3, 65 units to Shop4, and 85 units to Shop5. Traveling from
        Shop3, the costs are 87 units to Shop1, 15 units to Shop2, 45 units to
        Shop4, and 81 units to Shop5. From Shop4, it costs 76 units to go to
        Shop1, 65 units to Shop2, 45 units to Shop3, and 80 units to Shop5.
        Lastly, from Shop5, it takes 50 units to reach Shop1, 85 units to
        Shop2, 81 units to Shop3, and 80 units to Shop4.
        """
    )

    assert stocks.template_id == "stock_bond_balance_min_cost_ilp"
    assert math.isclose(stocks.objective_value or 0, 102000)
    assert services.template_id == "two_service_resource_min_cost_ilp"
    assert math.isclose(services.objective_value or 0, 20)
    assert projects.template_id == "environmental_project_three_var_min_cost_ilp"
    assert math.isclose(projects.objective_value or 0, 11340)
    assert properties.template_id == "property_pair_requirement_min_cost_ilp"
    assert math.isclose(properties.objective_value or 0, 1000000)
    assert tsp.template_id == "tsp_routing_enum"
    assert math.isclose(tsp.objective_value or 0, 203)


def test_template_solver_handles_phase68_nl4opt_models():
    transport = solve_with_template(
        """
        A concert organizer transports equipment using carts or trolleys.
        Carts can transport 5 kg/min of equipment and requires 2 workers.
        Trolleys can transport 7 kg/min of equipment and requires 4 workers.
        There must be at least 12 trolleys to be used. Only a maximum of 40%
        of the transportation can be using trolleys. The organizer has to
        deliver at a rate of 100 kg/min. Minimize the total number of workers.
        """
    )
    lights = solve_with_template(
        """
        A lighting company can install an LED fixture or a fluorescence lamp.
        The LED light uses 5 units of electricity per hour and needs to be
        changed 3 times a decade. The fluorescence lamp uses 8 units of
        electricity per hour and needs to be changed 4 times a decade. At
        least 30% implemented lights must be fluorescence lamps. The customer
        requires at least 300 light fixtures and can use at most 2000 units of
        electricity. Reduce the total number of light changes.
        """
    )
    cables = solve_with_template(
        """
        There is 1000 mg of gold available for long cables and short cables.
        Long cables require 10 mg of gold while short cables require 7 mg of
        gold. At least 5 times the number of short cables are needed than the
        long cables. There needs to be at least 10 long cables made. Each long
        cable sold results in a $12 profit and each short cable sold results
        in a $5 profit. Maximize profit.
        """
    )
    slicers = solve_with_template(
        """
        A butcher shop is buying manual and automatic slicers. The manual
        slicer can cut 5 slices per minute while the automatic slicer can cut
        8 slices per minute. The manual slicer requires 3 units of grease per
        minute while the automatic slicer requires 6 units of grease per
        minute. The number of manual slicers must be less than the number of
        automatic slicers. The shop needs to cut at least 50 slices per minute
        but can use at most 35 units of grease per minute. Minimize the total
        number of slicers.
        """
    )

    assert transport.template_id == "cart_trolley_worker_min_count_ilp"
    assert math.isclose(transport.objective_value or 0, 84)
    assert lights.template_id == "light_fixture_change_min_count_ilp"
    assert math.isclose(lights.objective_value or 0, 990)
    assert cables.template_id == "cable_mix_profit_max_ilp"
    assert math.isclose(cables.objective_value or 0, 819)
    assert slicers.template_id == "meat_slicer_min_count_ilp"
    assert math.isclose(slicers.objective_value or 0, -99999)


def test_template_solver_handles_phase69_industry_schedule_model():
    schedule = solve_with_template(
        r"""
        A project includes the following 7 activities, with their durations
        in days as follows: $A(4), B(3), C(5), D(2), E(10), F(10), G(1)$.
        The precedence relationships are also given as:
        $A \rightarrow G, D ; E, G \rightarrow F; D, F \rightarrow C ;
        F \rightarrow B$. The cost of work per day is 1000 Euros; additionally,
        a special machine must be rented from the start of activity $A$ to the
        end of activity $B$, costing 5000 Euros per day. Formulate this as a
        linear programming problem and solve it.
        """
    )

    assert schedule.template_id == "project_schedule_machine_rental_lp"
    assert math.isclose(schedule.objective_value or 0, 115000)


def test_template_solver_handles_phase69_mamo_models():
    shelf = solve_with_template(
        """
        A retail store manager is planning to allocate shelf space for two
        types of products: X and Y. The total shelf space available is limited
        to 2000 units. The combined shelf space of twice the amount for product
        X and product Y must be at least 500 units. The difference in shelf
        space between product X and twice the amount for product Y cannot
        exceed 1000 units. The cost associated with each unit of shelf space is
        $2 for product X and $3 for product Y. Minimize the total cost with
        integer shelf spaces.
        """
    )
    cargo = solve_with_template(
        """
        A shipping company allocates two types of cargo, $X$ and $Y$, to a ship.
        The total weight of both cargos combined cannot exceed 100 tons. The
        combined value calculated as twice the weight for cargo $X$ plus three
        times the weight for cargo $Y$, must be at least 150 units. The costs
        associated with each ton of cargo are $4 for cargo $X$ and $5 for cargo
        $Y. Minimize the total cost with whole tons.
        """
    )
    telecom = solve_with_template(
        """
        A telecommunications company is managing Project X and Project Y. The
        minimum total cost uses costs associated with each unit of resources
        allocated being $20 for project X and $30 for project Y. To ensure
        sufficient progress, five times the resources allocated to Project X
        combined with three times that of Project Y should at least be 60
        units. Four times the resources allocated to Project X along with
        those for Project Y should not exceed 80 units. The difference in
        resource allocation between Projects X and Y needs to be at least 10
        units in favor of Project X to maintain a strategic focus. Bounds are
        between 0 and 1000 units for project X and between 0 and 500 units for
        project Y.
        """
    )
    departments = solve_with_template(
        """
        A retail store manager is planning to allocate funds to four
        departments: $X1$, $X2$, $X3$, and $X4$. The costs associated with
        each unit of fund allocation are 2, 3, 4, and 5 units for departments
        $X1$, $X2$, $X3$ and $X4$ respectively. The combined funds allocated
        to departments $X1$ and $X2$ cannot exceed 2000 units. Twice the funds
        allocated for department $X3$ plus thrice that of department $X4$
        should be at least 3000 units. The difference in funds between
        department $X1$ and department $X4$ should not exceed 500 units.
        Department $X2$ should receive at least 1000 more units than department
        X3. Minimize total expenditure with whole-number allocations.
        """
    )
    supply = solve_with_template(
        """
        A supply chain manager is planning the allocation of resources across
        raw materials procurement, labor deployment, and transportation. The
        cost per unit of each resource is 4 for raw materials, 3 for labor,
        and 2 for transportation. The combined total of raw materials procured
        and labor deployed must be at least 50 units. The difference between
        the number of labor units and the number of transportation units
        should not exceed 10. The excess of raw material procurement over labor
        deployment should exactly equal to 20 units. Lastly, the combined total
        of raw materials procured and transportation cannot exceed 100 units.
        Minimize total cost with whole numbers.
        """
    )

    assert shelf.template_id == "shelf_space_balance_min_cost_ilp"
    assert math.isclose(shelf.objective_value or 0, 500)
    assert cargo.template_id == "cargo_value_capacity_min_cost_ilp"
    assert math.isclose(cargo.objective_value or 0, 250)
    assert telecom.template_id == "telecom_project_focus_min_cost_ilp"
    assert math.isclose(telecom.objective_value or 0, 240)
    assert departments.template_id == "four_department_fund_focus_min_cost_ilp"
    assert math.isclose(departments.objective_value or 0, 8000)
    assert supply.template_id == "supply_chain_resource_balance_min_cost_ilp"
    assert math.isclose(supply.objective_value or 0, 195)


def test_template_solver_handles_phase69_nl4opt_models():
    advertising = solve_with_template(
        """
        A cleaning company wants maximum exposure without exceeding their
        $250,000 advertising budget using radio ads and social media ads. Each
        radio ad costs $5,000; each social media ad costs $9,150. The expected
        exposure is 60,500 viewers for each radio ad. The expected exposure for
        each social media ad is 50,000 viewers. At least 15 but no more than
        40 radio ads should be ordered, and at least 35 social media ads should
        be contracted. How many ads should obtain maximum exposure?
        """
    )
    letters = solve_with_template(
        """
        A magic school sends letters by carrier pigeons or owls. A carrier
        pigeon can carry two letters at a time and requires 3 treats for
        service. An owl can carry 5 letters at a time and requires 5 treats for
        service. At most 40% of the birds can be owls. The school only has
        1000 treats available and at least 20 carrier pigeons must be used.
        Maximize the total number of letters.
        """
    )
    mountain = solve_with_template(
        """
        A tourist spot allows visitors to travel to the top either by hot-air
        balloon or gondola lift. A hot air balloon can carry 4 visitors while
        a gondola lift can carry 6 visitors. Each hot air balloon produces 10
        units of pollution while each gondola lift produces 15 units of
        pollution. There can be at most 10 hot-air balloon rides and at least
        70 visitors need to be transported. Minimize the total pollution
        produced.
        """
    )
    wagons = solve_with_template(
        """
        A mine sends ore to the surface in small and large wagons. A small
        wagon hold 20 units of ore while a large wagon holds 50 units of ore.
        The number of small wagons must be at least twice as much as the number
        or large wagons. At least 10 large wagons must be used. If 2000 units
        of ore need to taken to the surface, minimize the total number of
        wagons needed.
        """
    )

    assert advertising.template_id == "two_advertising_exposure_budget_ilp"
    assert advertising.status == "no_solution_reported"
    assert letters.template_id == "letter_bird_treat_max_ilp"
    assert math.isclose(letters.objective_value or 0, 841)
    assert mountain.template_id == "balloon_gondola_pollution_min_ilp"
    assert math.isclose(mountain.objective_value or 0, 175)
    assert wagons.template_id == "wagon_ore_min_count_ilp"
    assert math.isclose(wagons.objective_value or 0, 67)


def test_template_solver_handles_phase70_industry_costed_cutting_stock():
    result = solve_with_template(
        """
        A steel pipe retailer purchases raw pipes and cuts them according to
        customer requirements. Each raw steel pipe has a length of 1850 mm. A
        customer ordered 15 pieces of 290 mm, 28 pieces of 315 mm, 21 pieces
        of 350 mm, and 30 pieces of 455 mm. The number of cutting patterns to
        be used may not exceed four. Additional costs are incurred depending on
        the usage frequency of each cutting pattern: the most frequently used
        pattern incurs 1/10 of the value of one raw pipe; the second incurs
        2/10; the third incurs 3/10. Under each pattern, at most 5 pieces are
        produced from a single raw pipe. The leftover length for any cutting
        pattern may not exceed 100 mm. Minimize the total cost.
        """
    )

    assert result.template_id == "costed_pipe_cutting_stock_milp"
    assert math.isclose(result.objective_value or 0, 19.6)


def test_template_solver_handles_phase70_mamo_models():
    ships = solve_with_template(
        """
        A shipping company operates two types of cargo ships, X and Y. The
        company must transport at least 50 units of goods. The combined
        operation of 2 ships X and one ship Y cannot exceed 120 units.
        Operating a ship X costs 7 units while operating a ship Y costs 5
        units. Minimize total cost with whole-number ships.
        """
    )
    routes = solve_with_template(
        """
        A transportation company allocates its fleet of vehicles across three
        routes: X, Y, and Z. The operating cost per vehicle for routes X, Y,
        and Z are 10, 15, and 20 units respectively. The combined number of
        vehicles on routes X and Y cannot exceed 1000. The combined number of
        vehicles on routes Y and Z cannot exceed 800. The sum of vehicles on
        routes X and Z must be at least 500. Route X can accommodate up to 700
        vehicles; route Y can have a maximum of 600 vehicles; route Z can only
        handle up to 400 vehicles. Minimize total operating cost with whole
        vehicles.
        """
    )
    channels = solve_with_template(
        """
        A supply chain manager allocates resources across three distribution
        channels: X1, X2, and X3. The total resource allocation across all
        three channels cannot exceed 5000 units. Five times the resources
        allocated to it minus two and a half times the resources allocated to
        channel X2, of at least 1000 points, is required for X1. Three times
        the resources assigned to channel X2 plus four and a half times the
        resources assigned to channel X3, not exceeding 4000 points, is
        required. The difference between channel X1 and one and a half times
        that of channel X2 should be equal to X3. Channels x1, x2, and x3
        incurs costs of 10, 20, and 30 units respectively. Minimize total cost
        with integer allocations.
        """
    )
    education = solve_with_template(
        """
        An educational institution allocates resources for teacher salaries,
        classroom maintenance, textbook costs, and online tools. The combined
        resource allocation for teacher salaries and classroom maintenance
        should be at least 3000 units. The difference between textbook costs
        and online tools cannot exceed 500 units. The sum of all four
        allocations should not exceed a total of 5000 units. Teacher salaries
        should exceed classroom maintenance by at least 1000 units. Teacher
        salaries must be no less than 1500 units; classroom maintenance must
        be no less than 500 units; textbook costs must be no less than 200
        units; online tools must have at least a minimum investment of 100
        units. Funding costs are 5, 2, 3, and 1 respectively. Minimize total
        cost with whole numbers.
        """
    )

    assert ships.template_id == "two_ship_operation_min_cost_ilp"
    assert math.isclose(ships.objective_value or 0, 250)
    assert routes.template_id == "three_route_vehicle_allocation_min_cost_ilp"
    assert math.isclose(routes.objective_value or 0, 5000)
    assert channels.template_id == "three_channel_fractional_balance_min_cost_ilp"
    assert math.isclose(channels.objective_value or 0, 7000)
    assert education.template_id == "education_resource_min_cost_ilp"
    assert math.isclose(education.objective_value or 0, 12700)


def test_template_solver_handles_phase70_nl4opt_models():
    meal_delivery = solve_with_template(
        """
        A meal service company delivers meals either on electric bikes or
        scooters. A bike can hold 8 meals and requires 3 units of charge. A
        scooter can hold 5 meals and requires 2 units of charge. At most 30%
        of the electric vehicles can be bikes and at least 20 scooters must be
        used. The company only has 200 units of charge. Maximize the number of
        meals delivered.
        """
    )
    shoes = solve_with_template(
        """
        A shoe company supplies shoes to stores via vans and trucks. A van can
        transport 50 pairs of shoes while a truck can transport 100 pairs of
        shoes. The company must supply a minimum of 2000 pairs of shoes. The
        number of trucks used cannot exceed the number of vans used. Find the
        minimum number of vans that can be used.
        """
    )
    fruit = solve_with_template(
        """
        A food truck owner can spend at most $20000 on mangos and guavas. A
        mango costs the food truck owner $5 and a guava costs him $3. Each
        mango is sold for a profit of $3 while each guava is sold for a profit
        of $4. At least 100 mangos but at the most 150 are sold each month.
        The number of guavas sold is at most a third of the mangos sold.
        Maximize the profit.
        """
    )
    factories = solve_with_template(
        """
        A toy company can build a medium sized factory and a small factory. A
        medium sized factory can make 50 toys per day and requires 3 operators.
        A small factory can make 35 toys per day and requires 2 operators. The
        company must make at least 250 toys per day and only have available 16
        operators. Minimize the total number of factories.
        """
    )

    assert meal_delivery.template_id == "bike_scooter_meal_delivery_max_ilp"
    assert math.isclose(meal_delivery.objective_value or 0, 513)
    assert shoes.template_id == "van_truck_min_vans_ilp"
    assert math.isclose(shoes.objective_value or 0, 14)
    assert fruit.template_id == "mango_guava_profit_max_ilp"
    assert math.isclose(fruit.objective_value or 0, 650)
    assert factories.template_id == "two_factory_min_count_ilp"
    assert math.isclose(factories.objective_value or 0, 5)


def test_template_solver_handles_phase71_industry_dog_food_profit():
    result = solve_with_template(
        """
        Healthy Pet Foods Company produces two types of dog food: Meaties and
        Yummies. Each pack of Meaties contains 2 pounds of grains and 3 pounds
        of meat; each pack of Yummies contains 3 pounds of grains and 1.5
        pounds of meat. Meaties sell for $2.80 per pack, and Yummies sell for
        $2.00 per pack. A maximum of 400,000 pounds of grains can be purchased
        each month at a price of $0.20 per pound of grains. A maximum of
        300,000 pounds of meat can be purchased each month at a price of $0.50
        per pound of meat. A special machine is required to produce Meaties,
        with a monthly capacity of 90,000 packs. The variable costs for mixing
        and packaging dog food are $0.25 per pack (Meaties) and $0.20 per pack
        (Yummies). Maximize profit.
        """
    )

    assert result.template_id == "dog_food_profit_lp"
    assert math.isclose(result.objective_value or 0, 77500)


def test_template_solver_handles_phase71_mamo_models():
    education = solve_with_template(
        """
        An education department is planning to allocate resources between two
        programs, $X$ and $Y$. Each unit of resource allocated towards program
        $X$ yields a benefit score of 4, while each unit towards program $Y$
        yields a benefit score of 3. The objective is to minimize the total
        benefit score. The combined effort from twice the resources allocated
        to program $X$ and those allocated to program $Y$ must be at least 10
        units. The sum of the resources allocated to program $X$ and three
        times those allocated to program $Y$ cannot exceed 30 units. Allocations
        must be whole numbers.
        """
    )
    healthcare = solve_with_template(
        """
        A healthcare manager is planning the allocation of resources among
        three departments: X, Y, and Z. Each unit of resource allocated to these
        departments incurs a cost: 2 units for department X, 3 units for
        department Y, and 1 unit for department Z. The combined resources
        allocated to departments X and Y must be at least 50 units. The combined
        resources allocated to departments X and Z cannot exceed 100 units. The
        difference in resource allocation between departments Y and Z should be
        at least 10 units. What is the minimum total cost?
        """
    )
    military = solve_with_template(
        """
        A military operation requires the allocation of resources across three
        different units: Unit X1, Unit X2, and Unit X3. The total number of
        resources that can be allocated is limited to 100. Each unit has a cost
        associated with it, which is $50,000 for unit X1, $70,000 for unit X2,
        and $60,000 for unit X3. Half of the resources assigned to unit X1 plus
        six-tenths of those assigned to unit X3 should exceed or equal
        seven-tenths of the resources assigned to unit X2 by at least 20 units.
        Eight-tenths of the resources devoted to unit X2 and nine-tenths
        devoted to unit X3 should not surpass by more than 30 units what's been
        taken away from those dedicated to unit X1. Between 10 and 50 units can
        be allocated for Unit x1; between 20 and 70 units for Unit x2; and
        between zero and 40 units for Unit x3.
        """
    )
    environment = solve_with_template(
        """
        An environmental protection agency is planning to allocate its resources
        across four projects: $x1$, $x2$, $x3$, and $x4$. The costs per unit
        resource for projects $x1$, $x2$, $x3$ and $x4$ are 0.5 units, 0.3
        units, 0.4 units and 0.6 units respectively. The combined resources
        allocated for projects $x1$ and $x2$ cannot exceed 500 units. Twice the
        resources allocated for project $x1$ plus thrice that for project $x3$
        must be at least 300 units. Resources allocated for project $x4$ should
        not exceed half of those allocated for project $x2$ by more than 100
        units. The difference between resources allocated for project $x2$,
        plus those for project x4, minus those assigned to projects x1 and x3
        must be exactly equal to 80 units. $x1$: [0 ,400] $x2$: [0 ,300] $x3$:
        [50 ,200] $x4$: [30 ,60].
        """
    )

    assert education.template_id == "two_program_benefit_min_score_ilp"
    assert math.isclose(education.objective_value or 0, 20)
    assert healthcare.template_id == "three_department_resource_min_cost_ilp"
    assert math.isclose(healthcare.objective_value or 0, 110)
    assert military.template_id == "three_unit_fractional_military_min_cost_ilp"
    assert math.isclose(military.objective_value or 0, 4800000)
    assert environment.template_id == "four_project_environmental_resource_min_cost_ilp"
    assert math.isclose(environment.objective_value or 0, 103)


def test_template_solver_handles_phase71_nl4opt_models():
    pipes = solve_with_template(
        """
        The government is reworking the pipes to transport water to houses in
        the area. The water can be transported through wide pipes or narrow
        pipes. Wide pipes can transport 25 units of water per minute and narrow
        pipes can transport 15 units of water per minute. Due to logistics, the
        number of wide pipes can be at most a third the number of narrow pipes.
        If there needs to be at least 900 units of water transported every
        minute, and at least 5 wide pipes must be used, minimize the total
        number of pipes required.
        """
    )
    equipment = solve_with_template(
        """
        A handmade sports equipment manufacturing company makes basketballs and
        footballs. Basketballs require 5 units of materials and 1 hour to make
        whereas footballs require 3 units of materials and 2 hours to make. The
        company has available 1500 units of materials and workers can work for
        at most 750 hours. There must be at least three times as many
        basketballs as footballs and at least 50 footballs. Maximize the total
        number of sports equipment produced.
        """
    )
    desert = solve_with_template(
        """
        A company in the desert can transport goods to rural cities either by
        camel caravans or desert trucks. A camel caravan can deliver 50 units
        of goods per trip and takes 12 hours. A desert truck can deliver 150
        units of goods per trip and takes 5 hours. The company prefers to have
        more camel caravans than desert trucks. The company needs to deliver
        1500 units of goods, so minimize the total number of hours required.
        """
    )

    assert pipes.template_id == "wide_narrow_pipe_min_count_ilp"
    assert math.isclose(pipes.objective_value or 0, 52)
    assert equipment.template_id == "basketball_football_max_count_ilp"
    assert math.isclose(equipment.objective_value or 0, 333)
    assert desert.template_id == "camel_truck_transport_min_hours_ilp"
    assert math.isclose(desert.objective_value or 0, 136)


def test_template_solver_handles_phase72_mamo_resource_models():
    routes = solve_with_template(
        """
        A transportation manager is planning to distribute the transportation
        capacity among three routes: $X, Y$, and $Z$. The total capacity across
        all three routes cannot exceed 500 units. Route $X$ requires a minimum
        allocation of 100 units, route $Y$ needs at least 150 units, and route
        $Z$ requires a minimum allocation of 200 units. Each unit of capacity
        allocated to routes $X, Y$, and $Z$ incurs different costs, quantified
        as 5 , 10 , and 15 units respectively. Route X: between 0 and 300 units
        Route Y: between 0 and 400 units Route Z: between 0 and 600 units. What
        is the minimum total cost required?
        """
    )
    supply = solve_with_template(
        """
        A supply chain manager is planning the allocation of resources across
        three major areas: raw material procurement, labor, and transportation.
        Each unit of raw material, labor, and transportation incurs costs of 5,
        4, and 6 units respectively. The amount of raw materials should exceed
        twice the amount of labor by at least 500 units. The combined quantity
        of labor and transportation cannot exceed 1000 units. There must always
        be 200 more units in raw materials than there are in transportation.
        """
    )
    energy = solve_with_template(
        """
        In an energy sector, a company needs to allocate resources across four
        different projects: X1, X2, X3 and X4. The cost associated with each
        unit of investment in the projects are $\\$50$, $\\$60$, $\\$70$ and
        $\\$80$. The total investment in project X1 and project X2 cannot
        exceed 500 units. The total investment in project X3 and project X4
        cannot exceed 600 units. The investment difference between project (X3)
        and the combined investment of project (X1 + X2) should be at least 100
        units. Project (X4) must receive at least 50 more units of investment
        than project (X2). What would be the minimum possible total cost?
        """
    )

    assert routes.template_id == "three_route_capacity_lower_bound_min_cost_ilp"
    assert math.isclose(routes.objective_value or 0, 5000)
    assert supply.template_id == "supply_chain_material_labor_transport_min_cost_ilp"
    assert math.isclose(supply.objective_value or 0, 4300)
    assert energy.template_id == "four_energy_project_pair_margin_min_cost_ilp"
    assert math.isclose(energy.objective_value or 0, 11000)


def test_template_solver_handles_phase72_flow_and_diet_models():
    flow = solve_with_template(
        """
        Consider a complex railway network from the central hub to a key end
        destination. The capacity of each railway line is as follows:
        - From City 0 (Central Hub): Can dispatch passengers to City 1 (5,000
        passengers), and City 2 (7,000 passengers).
        - From City 1: Can dispatch passengers to City 2 (3,000 passengers).
        - From City 2 (End Destination): Can receive passengers from City 0
        (0 passengers) and City 1 (0 passengers).
        What is the maximum number of passengers that can be dispatched from
        the central hub to the end destination per day?
        """
    )
    diet = solve_with_template(
        """
        Bread contains 4 grams of protein, 7 grams of carbohydrates, has a
        calorie count of 130, and costs 3 dollar. Milk contains 6 grams of
        protein, 10 grams of carbohydrates, has a calorie count of 120, and
        costs 4 dollars. Fish is high in protein with 20 grams, contains no
        carbohydrates, has a calorie count of 150, and is the most expensive at
        8 dollars. Potato contains 1 gram of protein, is high in carbohydrates
        with 30 grams, has the lowest calorie count at 70, and is the cheapest
        at 2 dollars. The ideal intake for an adult is at least 40 grams of
        protein, 50 grams of carbohydrates, and 450 calories per day. Find the
        least costly way to achieve those amounts of nutrition.
        """
    )

    assert flow.template_id == "max_flow_network"
    assert math.isclose(flow.objective_value or 0, 10)
    assert diet.template_id == "narrative_food_diet_min_cost_lp"
    assert math.isclose(diet.objective_value or 0, 19.217560975609754)


def test_template_solver_handles_phase72_nl4opt_models():
    sanitizer = solve_with_template(
        """
        A company make both liquid and foam hand sanitizer. Liquid hand
        sanitizer requires 40 units of water and 50 units of alcohol. Foam hand
        sanitizer requires 60 units of water and 40 units of alcohol. The
        company has available 2000 units of water and 2100 units of alcohol.
        The number of foam hand sanitizers made must exceed the number of
        liquid hand sanitizers. In addition, at most 30 liquid hand sanitizers
        can be made. If each liquid hand sanitizer can clean 30 hands and each
        foam hand sanitizer can clean 20 hands, maximize the number of hands
        that can be cleaned.
        """
    )
    gummies = solve_with_template(
        """
        A boy needs to get enough magnesium and zinc in his diet by eating
        chewable gummies and taking pills. Each gummy contains 3 units of
        magnesium and 4 units of zinc. Each pill contains 2 units of magnesium
        and 5 units of zinc. The boy must take at least 10 pills. Since he
        prefers gummies more, he must eat at least 3 times the amount of
        gummies as pills. If the boy can consume at most 200 units of
        magnesium, maximize his zinc intake.
        """
    )
    chocolate = solve_with_template(
        """
        A chocolate company can transport their boxes of chocolate either using
        their own vans or by renting trucks. Their vans can transport 50 boxes
        per trip while a truck can transport 80 boxes per trip. The cost per
        van trip is $30 while the cost per truck trip is $50. The company needs
        to transport at least 1500 boxes of chocolate and they have a budget of
        $1000. The number of trips by van must be larger than the number of
        trips by trucks. Minimize the total number of trips.
        """
    )
    metal = solve_with_template(
        """
        A metal-working shop has access to two types of metal-working
        equipment, a chop saw and a steel cutter. A chop saw can work 25 pounds
        of steel and generates 25 units of waste every day. A steel cutter can
        only cut 5 pounds of steel and generates 3 units of waste every day.
        The shop must cut 520 pounds of metal every day and may at most produce
        400 units of waste every day. How should the shop purchase their
        equipment to decrease the total number of metal-working equipment
        needed?
        """
    )

    assert sanitizer.template_id == "sanitizer_cleaning_max_hands_ilp"
    assert math.isclose(sanitizer.objective_value or 0, 1000)
    assert gummies.template_id == "gummy_pill_zinc_max_ilp"
    assert math.isclose(gummies.objective_value or 0, 306)
    assert chocolate.template_id == "van_truck_chocolate_min_trips_ilp"
    assert math.isclose(chocolate.objective_value or 0, 24)
    assert metal.template_id == "metal_working_equipment_min_count_ilp"
    assert math.isclose(metal.objective_value or 0, 72)
