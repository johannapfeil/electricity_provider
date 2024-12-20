USE electricity_provider;

SELECT id, cons_12m
FROM client_data
GROUP BY id, cons_12m;

SELECT  p.id,
		c.channel_sales,

-- Compute time metrics of customers
		FLOOR(AVG(c.num_years_antig)) AS tenure_years,
        TIMESTAMPDIFF(MONTH, '2016-01-01', date_end) AS months_to_end,
        TIMESTAMPDIFF(MONTH, date_modif_prod, '2016-01-01') AS months_since_modified,
        TIMESTAMPDIFF(MONTH, date_renewal, '2016-01-01') AS months_since_renewed,
        
-- Compute yearly average consumption and margin metrics
		ROUND(AVG(c.cons_12m)) AS yearly_consumption,
        FLOOR(AVG(c.cons_last_month)) AS monthly_consumption,
        FLOOR(AVG(c.cons_gas_12m)) AS yearly_gas_consumption,
        ROUND(AVG(c.forecast_cons_12m)) AS yearly_forecast_consumption,
        ROUND(AVG(c.forecast_meter_rent_12m),3) AS yearly_forecast_meter_rent,
        ROUND(AVG(c.imp_cons),3) AS paid_consumption,
        ROUND(AVG(c.margin_gross_pow_ele),3) AS net_margin_electricity,
        ROUND(AVG(c.margin_net_pow_ele),3) AS net_margin_power,
		ROUND(AVG(c.net_margin),3) AS net_margin,
        
-- Compute yearly average price across periods for each customer
        ROUND(AVG(p.price_off_peak_var),3) AS avg_price_off_peak_var,
        ROUND(AVG(p.price_peak_var),3) AS avg_price_peak_var,
        ROUND(AVG(p.price_mid_peak_var),3) AS avg_price_mid_peak_var,

-- Compute yearly price ratios peak and off peak for each customer
		ROUND(AVG(p.price_peak_var / p.price_off_peak_var),3) AS price_peak_off_peak_ratio_var,

-- Compute yearly price variability (volatility)
		ROUND(STDDEV(p.price_off_peak_var),3) AS std_price_off_peak_var,
        ROUND(STDDEV(p.price_peak_var),3) AS std_price_peak_var,
        ROUND(STDDEV(p.price_mid_peak_var),3) AS std_price_mid_peak_var,

-- Compute price ranges across periods
		ROUND(MAX(p.price_peak_var) - MIN(p.price_peak_var),3) AS range_price_peak_var,
        ROUND(MAX(p.price_off_peak_var) - MIN(p.price_off_peak_var),3) AS range_price_off_peak_var,
        ROUND(MAX(p.price_mid_peak_var) - MIN(p.price_mid_peak_var),3) AS range_price_mid_peak_var,
        
		FLOOR(AVG(churn)) AS churn
                
FROM price_data AS p
JOIN client_data AS c
ON p.id = c.id
GROUP BY p.id, c.channel_sales, months_to_end, months_since_modified, months_since_renewed;










WITH cte_prices_prev AS (SELECT
		id, price_date,
		price_peak_var,
        price_off_peak_var,
        price_mid_peak_var,
        price_peak_fix,
        price_off_peak_fix,
        price_mid_peak_fix,
        
		LAG(price_peak_var) OVER(PARTITION BY id ORDER BY price_date) AS price_peak_var_prev,
        LAG(price_off_peak_var) OVER(PARTITION BY id ORDER BY price_date) AS price_off_peak_var_prev,
        LAG(price_mid_peak_var) OVER(PARTITION BY id ORDER BY price_date) AS price_mid_peak_var_prev,
        
        LAG(price_peak_fix) OVER(PARTITION BY id ORDER BY price_date) AS price_peak_fix_prev,
        LAG(price_off_peak_fix) OVER(PARTITION BY id ORDER BY price_date) AS price_off_peak_fix_prev,
        LAG(price_mid_peak_fix) OVER(PARTITION BY id ORDER BY price_date) AS price_mid_peak_fix_prev
FROM price_data
) SELECT id, price_date,

-- Compute price change relative to the last period 
		price_peak_var - price_peak_var_prev AS price_peak_var_change,
		price_off_peak_var - price_off_peak_var_prev AS price_off_peak_var_change,
		price_mid_peak_var - price_mid_peak_var_prev AS price_mid_peak_var_change,
            
		price_peak_fix - price_peak_fix_prev AS price_peak_fix_change,
		price_off_peak_fix - price_off_peak_fix_prev AS price_off_peak_fix_change,
		price_mid_peak_fix - price_mid_peak_fix_prev AS price_mid_peak_fix_change
FROM cte_prices_prev;