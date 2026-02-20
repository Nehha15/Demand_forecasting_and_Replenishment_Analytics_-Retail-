import pandas as pd
import numpy as np


def simulate_inventory(df, forecast_col,
                       lead_time=1,
                       service_level_z=1.65,
                       holding_cost=1.0,
                       stockout_cost=5.0):

    results = []

    # We simulate per Store + Dept
    grouped = df.groupby(["Store", "Dept"])

    for (store, dept), group in grouped:

        group = group.sort_values("Date").copy()

        # Estimate demand variability (std of historical demand)
        demand_std = group["Weekly_Sales"].std()

        safety_stock = service_level_z * demand_std * np.sqrt(lead_time)

        inventory = 0
        total_holding_cost = 0
        total_stockout_cost = 0
        total_demand = 0
        total_fulfilled = 0

        for _, row in group.iterrows():

            forecast = row[forecast_col]
            demand = row["Weekly_Sales"]

            # Order-up-to logic
            order_qty = max(forecast + safety_stock - inventory, 0)

            inventory += order_qty

            # Fulfill demand
            fulfilled = min(inventory, demand)
            stockout = max(demand - inventory, 0)

            inventory -= fulfilled

            # Cost calculations
            total_holding_cost += inventory * holding_cost
            total_stockout_cost += stockout * stockout_cost

            total_demand += demand
            total_fulfilled += fulfilled

        service_level = total_fulfilled / total_demand if total_demand > 0 else 0

        results.append({
            "Store": store,
            "Dept": dept,
            "HoldingCost": total_holding_cost,
            "StockoutCost": total_stockout_cost,
            "ServiceLevel": service_level
        })

    return pd.DataFrame(results)
