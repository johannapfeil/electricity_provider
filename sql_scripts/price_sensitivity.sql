USE electricity_provider;

-- Compute average price across periods
SELECT  id,
        AVG(price_off_peak_var) AS avg_price_off_peak_var
FROM price_data
GROUP BY id;
