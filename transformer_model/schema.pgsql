CREATE TABLE trade_records (
    id SERIAL PRIMARY KEY,
    ts_recv BIGINT NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_event_datetime TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    rtype SMALLINT NOT NULL,
    publisher_id INTEGER NOT NULL,
    instrument_id INTEGER NOT NULL,
    action CHAR(1) NOT NULL,
    side CHAR(1) NOT NULL,
    depth INTEGER NOT NULL,
    price_raw NUMERIC(19,0) NOT NULL,
    price NUMERIC(19,9) NOT NULL,
    size INTEGER NOT NULL,
    flags INTEGER NOT NULL,
    ts_in_delta INTEGER NOT NULL,
    sequence BIGINT NOT NULL
);

-- Create the instruments table
CREATE TABLE instruments (
    instrument_id INTEGER PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL
);

-- -- Update the foreign key constraint in the trade_records table
-- ALTER TABLE trade_records
-- DROP CONSTRAINT IF EXISTS fk_trade_records_instrument,
-- ADD CONSTRAINT fk_trade_records_instrument
-- FOREIGN KEY (instrument_id) REFERENCES instruments(instrument_id);

-- Indexes for trade_records table
CREATE INDEX idx_trade_records_instrument_id ON trade_records(instrument_id);
CREATE INDEX idx_trade_records_ts_event_datetime ON trade_records(ts_event_datetime);

-- Indexes for instruments table
CREATE INDEX idx_instruments_symbol ON instruments(symbol);
CREATE INDEX idx_instruments_start_date ON instruments(start_date);
CREATE INDEX idx_instruments_symbol_partial ON instruments(symbol) WHERE LENGTH(symbol) = 4;

-- Composite indexes for potential performance boost
CREATE INDEX idx_trade_records_composite ON trade_records(instrument_id, ts_event_datetime);
CREATE INDEX idx_instruments_composite ON instruments(instrument_id, symbol, start_date);

CREATE TABLE front_trades (
    id SERIAL PRIMARY KEY,
    ts_recv BIGINT NOT NULL,
    ts_event BIGINT NOT NULL,
    ts_event_datetime TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    rtype SMALLINT NOT NULL,
    publisher_id INTEGER NOT NULL,
    action CHAR(1) NOT NULL,
    side CHAR(1) NOT NULL,
    depth INTEGER NOT NULL,
    price_raw NUMERIC(19,0) NOT NULL,
    price NUMERIC(19,9) NOT NULL,
    size INTEGER NOT NULL,
    flags INTEGER NOT NULL,
    ts_in_delta INTEGER NOT NULL,
    sequence BIGINT NOT NULL,
    product_symbol VARCHAR(2) NOT NULL,
    contract_symbol VARCHAR(4) NOT NULL
);

create table public.candlestick_data
(
    id               integer default nextval('candlestick_data_id_seq1'::regclass) not null
        constraint candlestick_data_pkey1
            primary key,
    timestamp        timestamp                                                     not null,
    duration_seconds integer                                                       not null,
    product_symbol   varchar(2),
    contract_symbol  varchar(4),
    open_price       numeric(19, 9)                                                not null,
    high_price       numeric(19, 9)                                                not null,
    low_price        numeric(19, 9)                                                not null,
    close_price      numeric(19, 9)                                                not null,
    volume           integer                                                       not null,
    instrument_id    integer
);

alter table public.candlestick_data
    owner to postgres;

create index idx_candlestick_timestamp
    on public.candlestick_data (timestamp);

create index idx_candlestick_product_contract
    on public.candlestick_data (product_symbol, contract_symbol);

create index idx_candlestick_duration
    on public.candlestick_data (duration_seconds);


-- Function to determine the correct contract symbol for ES and NQ
CREATE OR REPLACE FUNCTION get_contract_symbol(product VARCHAR(2), trade_date DATE)
RETURNS VARCHAR(4) AS $$
DECLARE
    expiry_dates DATE[] := ARRAY[
        '2017-03-17'::DATE, '2017-06-16'::DATE, '2017-09-15'::DATE, '2017-12-15'::DATE,
        '2018-03-16'::DATE, '2018-06-15'::DATE, '2018-09-21'::DATE, '2018-12-21'::DATE,
        '2019-03-15'::DATE, '2019-06-21'::DATE, '2019-09-20'::DATE, '2019-12-20'::DATE,
        '2020-03-20'::DATE, '2020-06-19'::DATE, '2020-09-18'::DATE, '2020-12-18'::DATE,
        '2021-03-19'::DATE, '2021-06-18'::DATE, '2021-09-17'::DATE, '2021-12-17'::DATE,
        '2022-03-18'::DATE, '2022-06-17'::DATE, '2022-09-16'::DATE, '2022-12-16'::DATE,
        '2023-03-17'::DATE, '2023-06-16'::DATE, '2023-09-15'::DATE, '2023-12-15'::DATE,
        '2024-03-15'::DATE, '2024-06-21'::DATE, '2024-09-20'::DATE, '2024-12-20'::DATE
    ];
    year_code CHAR(1);
    month_code CHAR(1);
BEGIN
    year_code := RIGHT(EXTRACT(YEAR FROM trade_date)::TEXT, 1);

    -- Find the next expiry date
    FOR i IN 1..array_length(expiry_dates, 1) LOOP
        IF trade_date <= expiry_dates[i] THEN
            month_code := CASE EXTRACT(MONTH FROM expiry_dates[i])
                WHEN 3 THEN 'H'
                WHEN 6 THEN 'M'
                WHEN 9 THEN 'U'
                WHEN 12 THEN 'Z'
            END;
            RETURN product || month_code || year_code;
        END IF;
    END LOOP;

    -- If no expiry date is found (shouldn't happen with our data), return NULL
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Update query to populate contract_symbol and product_symbol in candlestick_data
UPDATE candlestick_data cd
SET
    contract_symbol = i.symbol,
    product_symbol = LEFT(i.symbol, 2)
FROM instruments i
WHERE cd.instrument_id = i.instrument_id AND LENGTH(i.symbol) = 4 AND i.symbol = get_contract_symbol(LEFT(i.symbol,2), cd.timestamp::date)

-- Main query to insert front month trades
INSERT INTO public.front_trades (
    ts_recv, ts_event, ts_event_datetime, rtype, publisher_id,
    action, side, depth, price_raw, price, size, flags,
    ts_in_delta, sequence, product_symbol, contract_symbol
)
SELECT
    tr.ts_recv, tr.ts_event, tr.ts_event_datetime, tr.rtype, tr.publisher_id,
    tr.action, tr.side, tr.depth, tr.price_raw, tr.price, tr.size, tr.flags,
    tr.ts_in_delta, tr.sequence,
    LEFT(i.symbol, 2) AS product_symbol,
    i.symbol AS contract_symbol
FROM
    public.trade_records tr
JOIN
    public.instruments i ON tr.instrument_id = i.instrument_id
WHERE
    EXTRACT(YEAR FROM tr.ts_event_datetime) IN (2023, 2024)
    AND LENGTH(i.symbol) = 4  -- Only contract trades, not spreads or options
    AND i.symbol = get_contract_symbol(LEFT(i.symbol, 2), tr.ts_event_datetime::DATE);



CREATE OR REPLACE FUNCTION generate_candlestick_data(p_duration_seconds INTEGER)
RETURNS void AS $$
BEGIN
    INSERT INTO candlestick_data (
        timestamp,
        duration_seconds,
        product_symbol,
        contract_symbol,
        open_price,
        high_price,
        low_price,
        close_price,
        volume,
        vwap
    )
    SELECT DISTINCT ON (time_bucket, product_symbol, contract_symbol)
        time_bucket,
        p_duration_seconds AS duration_seconds,
        product_symbol,
        contract_symbol,
        FIRST_VALUE(price) OVER w AS open_price,
        MAX(price) OVER w AS high_price,
        MIN(price) OVER w AS low_price,
        LAST_VALUE(price) OVER w AS close_price,
        SUM(size) OVER w AS volume,
        SUM(price * size) OVER w / NULLIF(SUM(size) OVER w, 0) AS vwap
    FROM (
        SELECT
            date_trunc(
                CASE
                    WHEN p_duration_seconds = 1 THEN 'second'
                    WHEN p_duration_seconds = 60 THEN 'minute'
                    WHEN p_duration_seconds = 3600 THEN 'hour'
                    WHEN p_duration_seconds = 86400 THEN 'day'
                END,
                ts_event_datetime
            ) AS time_bucket,
            product_symbol,
            contract_symbol,
            price,
            size
        FROM front_trades
    ) AS bucketed_trades
    WINDOW w AS (
        PARTITION BY time_bucket, product_symbol, contract_symbol
        ORDER BY time_bucket
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    )
    ORDER BY time_bucket, product_symbol, contract_symbol, time_bucket;
END;
$$ LANGUAGE plpgsql;

-- Generate 1-second candlesticks from raw data
SELECT public.generate_candlestick_data(1);

-- Generate 1-minute candlesticks from 1-second data
SELECT public.generate_candlestick_data(60);

-- Generate 1-hour candlesticks from 1-minute data
SELECT public.generate_candlestick_data(3600);

-- Generate 1-day candlesticks from 1-hour data
SELECT public.generate_candlestick_data(86400);

