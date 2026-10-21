{# Use the folder schema as-is (staging, core, marts) instead of dbt's default public_staging. #}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {{ custom_schema_name if custom_schema_name is not none else target.schema }}
{%- endmacro %}
