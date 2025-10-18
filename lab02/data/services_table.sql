DROP TABLE IF EXISTS services;
CREATE TABLE services (
    "Service:RDT-ID" INT,
    "Service:Date" DATE,
    "Service:Type" TEXT,
    "Service:Company" TEXT,
    "Service:Train number" INT,
    "Service:Completely cancelled" BOOLEAN,
    "Service:Partly cancelled" BOOLEAN,
    "Service:Maximum delay" INT,
    "Stop:RDT-ID" INT,
    "Stop:Station code" TEXT,
    "Stop:Station name" TEXT,
    "Stop:Arrival time" TIMESTAMP,
    "Stop:Arrival delay" INT,
    "Stop:Arrival cancelled" BOOLEAN,
    "Stop:Departure time" TIMESTAMP,
    "Stop:Departure delay" INT,
    "Stop:Departure cancelled" BOOLEAN,
    "Stop:Platform change" BOOLEAN,
    "Stop:Planned platform" TEXT,
    "Stop:Actual platform" TEXT
);
CREATE INDEX service_rdt_id ON services("Service:RDT-ID");
CREATE INDEX service_date ON services("Service:Date");