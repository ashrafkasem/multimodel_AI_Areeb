#!/bin/bash

echo "🔒 Security Configuration Test Script"
echo "Testing that vLLM models are only accessible through master API"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get local IP (for external access testing)
LOCAL_IP=$(hostname -I | awk '{print $1}')
if [ -z "$LOCAL_IP" ]; then
    LOCAL_IP=$(ip route get 8.8.8.8 | awk '{print $7; exit}')
fi

echo "🔍 Testing Configuration:"
echo "   • Local IP: $LOCAL_IP"
echo "   • Orchestrator: localhost:8000 (should be BLOCKED from outside)"
echo "   • Coder: localhost:8001 (should be BLOCKED from outside)"
echo "   • Master API: 0.0.0.0:8002 (should be ACCESSIBLE from outside)"
echo

# Test 1: Direct access to Orchestrator (should fail from external IP)
echo "Test 1: Testing external access to Orchestrator (port 8000)"
echo -n "Trying http://$LOCAL_IP:8000/health ... "
if curl -s --connect-timeout 5 "http://$LOCAL_IP:8000/health" > /dev/null 2>&1; then
    echo -e "${RED}❌ SECURITY RISK: Orchestrator accessible from outside!${NC}"
    SECURITY_ISSUE=1
else
    echo -e "${GREEN}✅ SECURE: Orchestrator blocked from external access${NC}"
fi

# Test 2: Direct access to Coder (should fail from external IP)
echo "Test 2: Testing external access to Coder (port 8001)"
echo -n "Trying http://$LOCAL_IP:8001/health ... "
if curl -s --connect-timeout 5 "http://$LOCAL_IP:8001/health" > /dev/null 2>&1; then
    echo -e "${RED}❌ SECURITY RISK: Coder accessible from outside!${NC}"
    SECURITY_ISSUE=1
else
    echo -e "${GREEN}✅ SECURE: Coder blocked from external access${NC}"
fi

# Test 3: Localhost access to Orchestrator (should work)
echo "Test 3: Testing localhost access to Orchestrator (port 8000)"
echo -n "Trying http://localhost:8000/health ... "
if curl -s --connect-timeout 5 "http://localhost:8000/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ OK: Orchestrator accessible from localhost${NC}"
else
    echo -e "${YELLOW}⚠️  WARNING: Orchestrator not responding on localhost${NC}"
fi

# Test 4: Localhost access to Coder (should work)
echo "Test 4: Testing localhost access to Coder (port 8001)"
echo -n "Trying http://localhost:8001/health ... "
if curl -s --connect-timeout 5 "http://localhost:8001/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ OK: Coder accessible from localhost${NC}"
else
    echo -e "${YELLOW}⚠️  WARNING: Coder not responding on localhost${NC}"
fi

# Test 5: External access to Master API (should work)
echo "Test 5: Testing external access to Master API (port 8002)"
echo -n "Trying http://$LOCAL_IP:8002/health ... "
if curl -s --connect-timeout 5 "http://$LOCAL_IP:8002/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ OK: Master API accessible from outside${NC}"
else
    echo -e "${YELLOW}⚠️  WARNING: Master API not responding${NC}"
fi

# Test 6: Check listening ports
echo
echo "Test 6: Checking listening ports"
echo "Port status:"
netstat -tlnp 2>/dev/null | grep -E ":(8000|8001|8002)" | while read line; do
    if echo "$line" | grep -q "127.0.0.1:8000"; then
        echo -e "   ${GREEN}✅ Port 8000: Bound to localhost only (SECURE)${NC}"
    elif echo "$line" | grep -q "0.0.0.0:8000\|:::8000"; then
        echo -e "   ${RED}❌ Port 8000: Bound to all interfaces (SECURITY RISK)${NC}"
        SECURITY_ISSUE=1
    elif echo "$line" | grep -q "127.0.0.1:8001"; then
        echo -e "   ${GREEN}✅ Port 8001: Bound to localhost only (SECURE)${NC}"
    elif echo "$line" | grep -q "0.0.0.0:8001\|:::8001"; then
        echo -e "   ${RED}❌ Port 8001: Bound to all interfaces (SECURITY RISK)${NC}"
        SECURITY_ISSUE=1
    elif echo "$line" | grep -q "0.0.0.0:8002\|:::8002"; then
        echo -e "   ${GREEN}✅ Port 8002: Bound to all interfaces (CORRECT for Master API)${NC}"
    fi
done

echo
if [ "${SECURITY_ISSUE:-0}" = "1" ]; then
    echo -e "${RED}🚨 SECURITY ISSUES DETECTED!${NC}"
    echo "❌ Your vLLM models are accessible from outside"
    echo "📝 To fix this:"
    echo "   1. Stop current vLLM processes: pkill -f 'vllm serve'"
    echo "   2. Use secure script: ./start_vllm_secure.sh"
    echo "   3. Re-run this test: ./test_security.sh"
else
    echo -e "${GREEN}🔒 SECURITY CONFIGURATION CORRECT!${NC}"
    echo "✅ vLLM models are only accessible from localhost"
    echo "✅ External access is only through Master API on port 8002"
    echo "✅ Your setup is secure!"
fi

echo
echo "📋 Summary:"
echo "   • Orchestrator (port 8000): Should be localhost-only"
echo "   • Coder (port 8001): Should be localhost-only"  
echo "   • Master API (port 8002): Should accept external connections"
echo "   • Use API keys for additional security on port 8002" 