# Cortex Sandbox Container
# This container provides a sandboxed environment for MCP tool execution

FROM python:3.12-slim

# Install common development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages needed for MCP servers
RUN pip install --no-cache-dir \
    fastmcp>=2.0.0

# Create workspace directory
WORKDIR /workspace

# Copy MCP server files
COPY mcp_servers/ /opt/mcp_servers/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command - keep container running
CMD ["tail", "-f", "/dev/null"]
