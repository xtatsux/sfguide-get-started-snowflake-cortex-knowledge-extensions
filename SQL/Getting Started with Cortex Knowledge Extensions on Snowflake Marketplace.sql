USE ROLE accountadmin;
USE SCHEMA shared;
USE WAREHOUSE compute_wh;

DESCRIBE CORTEX SEARCH SERVICE cke_snowflake_docs_service;

select 
    snowflake.cortex.search_preview(
        'snowflake_documentation.shared.cke_snowflake_docs_service', 
        '{ "query": "What is a table in Snowflake?", "columns": ["chunk","document_title", "source_url"] }');


USE ROLE accountadmin;
USE WAREHOUSE compute_wh;

CREATE OR REPLACE DATABASE chatbot_db;
CREATE OR REPLACE SCHEMA chatbot_schema;

USE ROLE ACCOUNTADMIN;
ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'ANY_REGION';