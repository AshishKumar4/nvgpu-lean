# The Lean memory model agrees with Lustig et al.'s Alloy model of the mixed-proxy PTX memory model
# (NVlabs/mixedproxy at the commit tools/litmus/crosscheck.py pins; Alloy 5.1.0, pinned by hash)
# read with fences as generic-proxy operations, on every nvlitmus test of the artifact and of
# tools/litmus/tests.
MIXEDPROXY_SHA=$(sed -n 's/^MIXEDPROXY_SHA = "\(.*\)"$/\1/p' tools/litmus/crosscheck.py)
ALLOY_SHA256=a3b43e8ec9967947aea2d5101bd96b3c4eb0d81eb3dc9bba41cc9649349c690a
dir=$(mktemp -d)
git clone -q https://github.com/NVlabs/mixedproxy "$dir/mixedproxy"
git -C "$dir/mixedproxy" checkout -q "$MIXEDPROXY_SHA"
jar="$dir/mixedproxy/alloy/org.alloytools.alloy.dist.jar"
curl -fsSL -o "$jar" https://github.com/AlloyTools/org.alloytools.alloy/releases/download/v5.1.0/org.alloytools.alloy.dist.jar
echo "$ALLOY_SHA256  $jar" | sha256sum -c
lake build litmus
python3 tools/litmus/crosscheck.py --mixedproxy "$dir/mixedproxy" --jobs 8
