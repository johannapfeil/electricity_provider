USE electricity_provider;

-- Disable foreign key checks to prevent errors when dropping tables
SET FOREIGN_KEY_CHECKS = 0;
-- Drop tables if they already exist
DROP TABLE IF EXISTS client_data;
DROP TABLE IF EXISTS price_data;

-- Re-enable foreign key checks
SET FOREIGN_KEY_CHECKS = 1;

CREATE TABLE `client_data` (
    `id` VARCHAR(32) NOT NULL PRIMARY KEY,
    `sales_channel` VARCHAR(32) NULL,
    `consumption_12months` FLOAT NULL,
    `consumption_gas_12months` FLOAT NULL,
    `consumption_last_month` FLOAT NULL,
    `date_activated` DATE NOT NULL,
    `date_end` DATE NULL,
    `date_last_modification` DATE NULL,
    `date_renewal` DATE NULL,
    `forecast_consumption_12months` FLOAT NULL,
    `forecast_consumption_year` FLOAT NULL,
    `forecast_discount_energy` FLOAT NULL,
    `forecast_meter_rent_12months` FLOAT NULL,
    `forecast_price_energy_off_peak` FLOAT NULL,
    `forecast_price_energy_peak` FLOAT NULL,
    `forecast_price_power_off_peak` FLOAT NULL,
    `has_gas` BOOLEAN NOT NULL,
    `paid_consumption` FLOAT NULL,
    `margin_gross_power` FLOAT NULL,
    `margin_net_power` FLOAT NULL,
    `num_products_active` INT NOT NULL,
    `net_margin` FLOAT NULL,
    `num_years_antiquity` INT NULL,
    `original_subscription` VARCHAR(32) NULL,
    `power` FLOAT NULL,
    `churn` BOOLEAN NOT NULL
);

CREATE TABLE `price_data` (
    `id` CHAR(32) NOT NULL,
    `price_date` DATE NOT NULL PRIMARY KEY,
    `price_off_peak_variable` FLOAT NULL,
    `price_peak_variable` FLOAT NULL,
    `price_mid_peak_variable` FLOAT NULL,
    `price_off_peak_fixed` FLOAT NULL,
    `price_peak_fixed` FLOAT NULL,
    `price_mid_peak_fixed` FLOAT NULL,
    
FOREIGN KEY (`id`) REFERENCES `client_data`(`id`)
);