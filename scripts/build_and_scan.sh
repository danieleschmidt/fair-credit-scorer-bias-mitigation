#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="fair-credit-scorer"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD)
VERSION=${VERSION:-$(git describe --tags --always --dirty)}

echo "🏗️  Building and scanning Fair Credit Scorer container image"
echo "=========================================================="
echo "Image: $IMAGE_NAME"
echo "Version: $VERSION"
echo "Build Date: $BUILD_DATE"
echo "VCS Ref: $VCS_REF"
echo ""

cd "$PROJECT_ROOT"

check_dependencies() {
    echo "🔍 Checking dependencies..."
    
    local missing_deps=()
    
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi
    
    if ! command -v trivy &> /dev/null; then
        missing_deps+=("trivy")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        echo "❌ Missing dependencies: ${missing_deps[*]}"
        echo "Please install the missing dependencies and try again."
        exit 1
    fi
    
    echo "✅ All dependencies found"
}

build_image() {
    local target="${1:-production}"
    local tag="${IMAGE_NAME}:${VERSION}-${target}"
    
    echo "🔨 Building $target image: $tag"
    
    docker build \
        --target "$target" \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VCS_REF="$VCS_REF" \
        --build-arg VERSION="$VERSION" \
        --tag "$tag" \
        --label "org.opencontainers.image.title=Fair Credit Scorer" \
        --label "org.opencontainers.image.description=Fair Credit Scoring with Bias Mitigation" \
        --label "org.opencontainers.image.version=$VERSION" \
        --label "org.opencontainers.image.created=$BUILD_DATE" \
        --label "org.opencontainers.image.revision=$VCS_REF" \
        --label "org.opencontainers.image.vendor=Terragon Labs" \
        --label "org.opencontainers.image.licenses=MIT" \
        --label "org.opencontainers.image.source=https://github.com/danieleschmidt/fair-credit-scorer-bias-mitigation" \
        .
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully built $tag"
        return 0
    else
        echo "❌ Failed to build $tag"
        return 1
    fi
}

scan_image() {
    local target="${1:-production}"
    local tag="${IMAGE_NAME}:${VERSION}-${target}"
    
    echo "🔍 Scanning image for vulnerabilities: $tag"
    
    local report_file="trivy-report-${target}.json"
    
    trivy image \
        --format json \
        --output "$report_file" \
        --exit-code 1 \
        --severity HIGH,CRITICAL \
        "$tag"
    
    local scan_result=$?
    
    if [ $scan_result -eq 0 ]; then
        echo "✅ Security scan passed for $tag"
        echo "📄 Report saved to: $report_file"
    else
        echo "❌ Security scan failed for $tag"
        echo "📄 Report saved to: $report_file"
        
        echo ""
        echo "High and Critical vulnerabilities found:"
        trivy image --format table --severity HIGH,CRITICAL "$tag"
    fi
    
    return $scan_result
}

test_image() {
    local target="${1:-production}"
    local tag="${IMAGE_NAME}:${VERSION}-${target}"
    
    echo "🧪 Testing image: $tag"
    
    local container_name="test-${IMAGE_NAME}-${target}-$$"
    
    echo "Starting container: $container_name"
    docker run -d --name "$container_name" "$tag" sleep 60
    
    local container_id
    container_id=$(docker ps -qf "name=$container_name")
    
    if [ -z "$container_id" ]; then
        echo "❌ Failed to start container"
        return 1
    fi
    
    echo "Running health checks..."
    
    local test_passed=true
    
    if ! docker exec "$container_name" python -c "import fair_credit_scorer_bias_mitigation; print('✅ Package import successful')"; then
        echo "❌ Package import failed"
        test_passed=false
    fi
    
    if ! docker exec "$container_name" python -m pytest --version > /dev/null 2>&1; then
        echo "❌ pytest not available in production image"
    else
        echo "✅ Development tools available"
    fi
    
    echo "Cleaning up test container..."
    docker stop "$container_name" > /dev/null 2>&1 || true
    docker rm "$container_name" > /dev/null 2>&1 || true
    
    if [ "$test_passed" = true ]; then
        echo "✅ Image tests passed"
        return 0
    else
        echo "❌ Image tests failed"
        return 1
    fi
}

generate_sbom() {
    local target="${1:-production}"
    local tag="${IMAGE_NAME}:${VERSION}-${target}"
    
    echo "📋 Generating SBOM for: $tag"
    
    local sbom_file="sbom-${target}.json"
    
    if command -v syft &> /dev/null; then
        syft "$tag" -o spdx-json > "$sbom_file"
        echo "✅ SBOM generated: $sbom_file"
    else
        echo "⚠️  Syft not found, skipping SBOM generation"
        echo "   Install Syft to generate Software Bill of Materials"
    fi
}

sign_image() {
    local target="${1:-production}"
    local tag="${IMAGE_NAME}:${VERSION}-${target}"
    
    echo "✍️  Signing image: $tag"
    
    if command -v cosign &> /dev/null && [ -n "${COSIGN_PRIVATE_KEY:-}" ]; then
        echo "Signing with Cosign..."
        cosign sign --key env://COSIGN_PRIVATE_KEY "$tag"
        echo "✅ Image signed successfully"
    else
        echo "⚠️  Cosign not configured, skipping image signing"
        echo "   Set COSIGN_PRIVATE_KEY environment variable to enable signing"
    fi
}

push_image() {
    local target="${1:-production}"
    local tag="${IMAGE_NAME}:${VERSION}-${target}"
    local registry="${REGISTRY:-}"
    
    if [ -z "$registry" ]; then
        echo "⚠️  No registry specified, skipping push"
        echo "   Set REGISTRY environment variable to enable pushing"
        return 0
    fi
    
    local full_tag="${registry}/${tag}"
    
    echo "📤 Pushing image: $full_tag"
    
    docker tag "$tag" "$full_tag"
    docker push "$full_tag"
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully pushed $full_tag"
    else
        echo "❌ Failed to push $full_tag"
        return 1
    fi
}

cleanup() {
    echo "🧹 Cleaning up dangling images..."
    docker image prune -f
}

show_help() {
    cat << EOF
Build and Security Scanning Script for Fair Credit Scorer

Usage: $0 [OPTIONS] [COMMAND]

Commands:
    build [TARGET]     Build image (default: production)
    scan [TARGET]      Scan image for vulnerabilities  
    test [TARGET]      Test image functionality
    sbom [TARGET]      Generate Software Bill of Materials
    sign [TARGET]      Sign image with Cosign
    push [TARGET]      Push image to registry
    all [TARGET]       Run complete build pipeline
    clean              Clean up dangling images

Targets:
    development        Development image with tools
    production         Production-optimized image (default)

Options:
    -h, --help         Show this help message
    -v, --version      Show version information

Environment Variables:
    VERSION           Image version (default: git describe)
    REGISTRY          Container registry for pushing
    COSIGN_PRIVATE_KEY Private key for image signing

Examples:
    $0 build                    # Build production image
    $0 build development        # Build development image
    $0 scan production          # Scan production image
    $0 all                      # Complete pipeline for production
    REGISTRY=ghcr.io/user $0 push # Push to GitHub Container Registry

EOF
}

main() {
    local command="${1:-all}"
    local target="${2:-production}"
    
    case "$command" in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--version)
            echo "$VERSION"
            exit 0
            ;;
        build)
            check_dependencies
            build_image "$target"
            ;;
        scan)
            check_dependencies
            scan_image "$target"
            ;;
        test)
            check_dependencies
            test_image "$target"
            ;;
        sbom)
            generate_sbom "$target"
            ;;
        sign)
            sign_image "$target"
            ;;
        push)
            push_image "$target"
            ;;
        clean)
            cleanup
            ;;
        all)
            echo "🚀 Running complete build pipeline for $target target"
            echo ""
            
            check_dependencies
            
            if ! build_image "$target"; then
                echo "❌ Build failed, stopping pipeline"
                exit 1
            fi
            
            echo ""
            if ! test_image "$target"; then
                echo "❌ Tests failed, stopping pipeline"
                exit 1
            fi
            
            echo ""
            generate_sbom "$target"
            
            echo ""
            local scan_failed=false
            if ! scan_image "$target"; then
                echo "⚠️  Security scan failed, but continuing..."
                scan_failed=true
            fi
            
            echo ""
            sign_image "$target"
            
            echo ""
            push_image "$target"
            
            echo ""
            cleanup
            
            echo ""
            echo "🎉 Build pipeline completed!"
            echo "Image: ${IMAGE_NAME}:${VERSION}-${target}"
            
            if [ "$scan_failed" = true ]; then
                echo "⚠️  Note: Security vulnerabilities were found"
                exit 1
            fi
            ;;
        *)
            echo "❌ Unknown command: $command"
            echo "Use '$0 --help' for usage information"
            exit 1
            ;;
    esac
}

main "$@"