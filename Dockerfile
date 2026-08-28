# Railway build for the persistent, ciocore-capable MCP host.
#
# Why a Dockerfile at all:
#
# Railway now builds with **railpack**, which ignores both railway.toml and
# nixpacks.toml — so the `pip install ciocore` phase nixpacks.toml was written to
# add never ran. Railpack does defer to a Dockerfile when one is present, so this
# is the only place the install can actually be controlled. It also ends the
# nixpacks/railpack config drift for good.
#
# Why ciocore is installed separately, with --no-deps:
#
#   ciocore 9.1.1 (and 9.2.0) hard-pin  pyjwt==2.9.0
#   mcp     1.27.1            declares  pyjwt>=2.10.1
#
# pip therefore refuses the combination outright ("ResolutionImpossible"). But
# the conflict is DECLARED, not real: the local venv that successfully submitted
# Conductor job 00005 on 2026-08-27 runs mcp 1.27.1 against pyjwt 2.9.0. So this
# reproduces that exact, proven combination rather than guessing which side to
# bend — install normally, add ciocore without letting it drag its pin in, then
# put pyjwt back to the version both are known to work with.
#
# requirements.txt is deliberately left WITHOUT ciocore so Vercel's serverless
# bundle stays lean; conductor_render.py imports ciocore lazily, so that host
# keeps serving the read-only tools. Railway gets the full stack from here.

FROM python:3.13-slim

WORKDIR /app

# Build deps for any wheel that needs compiling, removed in the same layer.
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir --no-deps ciocore==9.1.1 \
 && pip install --no-cache-dir requests "cioseq>=0.4.1,<1.0.0" "click>=8.1.3,<9.0.0" \
      "markdown>=3.5.2,<4.0.0" "colorlog>=6.8.2,<7.0.0" \
 && pip install --no-cache-dir --no-deps --force-reinstall "pyjwt==2.9.0"

# Fail the BUILD, not the first submission, if the balancing act above breaks.
RUN python -c "import jwt, mcp, ciocore.conductor_submit; print('ciocore + mcp import OK, pyjwt', jwt.__version__)"

COPY . .

# server.py defaults to stdio; sse is what opens an HTTP port. Set here as well
# as in the Railway service vars so the image is correct on its own.
ENV MCP_TRANSPORT=sse \
    STATELESS_HTTP=1 \
    PORT=8080

EXPOSE 8080
CMD ["python", "server.py"]
