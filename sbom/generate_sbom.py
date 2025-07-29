#!/usr/bin/env python3
"""
Advanced SBOM Generation for MATURING repositories
Software Bill of Materials generation with Syft for supply chain transparency
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import hashlib


class SBOMGenerator:
    """Advanced SBOM generation with supply chain security analysis."""
    
    def __init__(self, project_name: str = "fair-credit-scorer-bias-mitigation"):
        self.project_name = project_name
        self.project_root = Path(__file__).parent.parent
        self.sbom_dir = self.project_root / "sbom_reports"
        self.sbom_dir.mkdir(exist_ok=True)
        
        # SBOM formats to generate
        self.formats = ["json", "spdx-json", "cyclonedx-json", "table"]
        
        # Syft configuration
        self.syft_config = {
            "catalogers": {
                "enabled": [
                    "python-installed-package-cataloger",
                    "python-package-cataloger",
                    "python-poetry-lock-cataloger",
                    "python-pipfile-lock-cataloger",
                    "python-pip-requirements-cataloger"
                ]
            },
            "package": {
                "search-unindexed-archives": True,
                "search-indexed-archives": True
            },
            "file-metadata": {
                "enabled": True,
                "digests": ["sha256", "sha1"]
            }
        }
    
    def generate_comprehensive_sbom(self, include_dev_deps: bool = True,
                                   include_container: bool = True) -> Dict[str, Any]:
        """Generate comprehensive SBOM with multiple formats and analysis."""
        print("📋 Starting comprehensive SBOM generation...")
        
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "project_name": self.project_name,
            "formats_generated": [],
            "analysis": {},
            "security_summary": {},
            "supply_chain_risks": []
        }
        
        # Step 1: Generate SBOM in multiple formats
        sbom_files = self._generate_multi_format_sbom(include_dev_deps)
        results["formats_generated"] = list(sbom_files.keys())
        
        # Step 2: Analyze dependencies
        if "json" in sbom_files:
            analysis = self._analyze_dependencies(sbom_files["json"])
            results["analysis"] = analysis
        
        # Step 3: Generate container SBOM if requested
        if include_container:
            container_sbom = self._generate_container_sbom()
            if container_sbom:
                results["container_sbom"] = container_sbom
        
        # Step 4: Security analysis
        security_summary = self._perform_security_analysis(sbom_files)
        results["security_summary"] = security_summary
        
        # Step 5: Supply chain risk assessment
        risks = self._assess_supply_chain_risks(results["analysis"])
        results["supply_chain_risks"] = risks
        
        # Step 6: Generate comprehensive report
        report_path = self._generate_comprehensive_report(results)
        results["report_path"] = report_path
        
        return results
    
    def _generate_multi_format_sbom(self, include_dev_deps: bool = True) -> Dict[str, str]:
        """Generate SBOM in multiple formats using Syft."""
        print("🔍 Generating SBOM in multiple formats...")
        
        sbom_files = {}
        
        # Determine source directory
        source_path = str(self.project_root)
        
        for fmt in self.formats:
            try:
                print(f"  Generating {fmt} format...")
                
                # Determine output file extension
                if fmt == "table":
                    ext = "txt"
                elif fmt == "spdx-json":
                    ext = "spdx.json"
                elif fmt == "cyclonedx-json":
                    ext = "cyclonedx.json"
                else:
                    ext = fmt
                
                output_file = self.sbom_dir / f"sbom.{ext}"
                
                # Build syft command
                cmd = [
                    "syft",
                    "packages",
                    source_path,
                    "-o", fmt,
                    "--file", str(output_file)
                ]
                
                # Add configuration for detailed analysis
                if fmt in ["json", "spdx-json", "cyclonedx-json"]:
                    cmd.extend([
                        "--catalogers", "python-installed-package-cataloger,python-package-cataloger",
                        "--scope", "all-layers" if include_dev_deps else "squashed"
                    ])
                
                # Execute syft
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=self.project_root
                )
                
                if result.returncode == 0 and output_file.exists():
                    sbom_files[fmt] = str(output_file)
                    print(f"  ✅ {fmt} SBOM generated: {output_file}")
                else:
                    print(f"  ❌ Failed to generate {fmt} SBOM")
                    print(f"     Error: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print(f"  ⏰ {fmt} SBOM generation timed out")
            except Exception as e:
                print(f"  💥 Error generating {fmt} SBOM: {e}")
        
        return sbom_files
    
    def _generate_container_sbom(self) -> Optional[Dict[str, Any]]:
        """Generate SBOM for Docker container if Dockerfile exists."""
        dockerfile_path = self.project_root / "Dockerfile"
        
        if not dockerfile_path.exists():
            print("  ⚠️ No Dockerfile found, skipping container SBOM")
            return None
        
        print("🐳 Generating container SBOM...")
        
        try:
            # Build container image first
            image_name = f"{self.project_name}:sbom-analysis"
            
            build_cmd = [
                "docker", "build",
                "-t", image_name,
                "-f", str(dockerfile_path),
                "."
            ]
            
            print("  Building container image for SBOM analysis...")
            build_result = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=self.project_root
            )
            
            if build_result.returncode != 0:
                print(f"  ❌ Container build failed: {build_result.stderr}")
                return None
            
            # Generate container SBOM
            container_sbom_file = self.sbom_dir / "container_sbom.json"
            
            syft_cmd = [
                "syft",
                "packages",
                image_name,
                "-o", "json",
                "--file", str(container_sbom_file)
            ]
            
            result = subprocess.run(
                syft_cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0 and container_sbom_file.exists():
                print(f"  ✅ Container SBOM generated: {container_sbom_file}")
                
                # Clean up image
                subprocess.run(["docker", "rmi", image_name], capture_output=True)
                
                return {
                    "file_path": str(container_sbom_file),
                    "image_name": image_name,
                    "layers_analyzed": True
                }
            else:
                print(f"  ❌ Container SBOM generation failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("  ⏰ Container SBOM generation timed out")
            return None
        except Exception as e:
            print(f"  💥 Container SBOM error: {e}")
            return None
    
    def _analyze_dependencies(self, sbom_file: str) -> Dict[str, Any]:
        """Analyze dependencies from JSON SBOM."""
        print("🔬 Analyzing dependencies...")
        
        try:
            with open(sbom_file, 'r') as f:
                sbom_data = json.load(f)
            
            artifacts = sbom_data.get("artifacts", [])
            
            # Categorize dependencies
            analysis = {
                "total_packages": len(artifacts),
                "by_type": {},
                "by_ecosystem": {},
                "licenses": {},
                "versions": {},
                "top_level_deps": [],
                "transitive_deps": [],
                "outdated_packages": [],
                "security_critical": []
            }
            
            # Analyze each artifact
            for artifact in artifacts:
                pkg_type = artifact.get("type", "unknown")
                ecosystem = artifact.get("language", "unknown")
                name = artifact.get("name", "unknown")
                version = artifact.get("version", "unknown")
                licenses = artifact.get("licenses", [])
                
                # Count by type
                analysis["by_type"][pkg_type] = analysis["by_type"].get(pkg_type, 0) + 1
                
                # Count by ecosystem
                analysis["by_ecosystem"][ecosystem] = analysis["by_ecosystem"].get(ecosystem, 0) + 1
                
                # Collect licenses
                for license_info in licenses:
                    license_name = license_info if isinstance(license_info, str) else license_info.get("value", "unknown")
                    analysis["licenses"][license_name] = analysis["licenses"].get(license_name, 0) + 1
                
                # Collect version info
                if name != "unknown" and version != "unknown":
                    analysis["versions"][name] = version
                
                # Identify top-level vs transitive dependencies
                # This is simplified - in practice, you'd need dependency tree analysis
                if pkg_type == "python" and ecosystem == "python":
                    if self._is_top_level_dependency(name):
                        analysis["top_level_deps"].append({"name": name, "version": version})
                    else:
                        analysis["transitive_deps"].append({"name": name, "version": version})
                
                # Flag security-critical packages
                if name.lower() in ["cryptography", "pycryptodome", "requests", "urllib3", "pillow"]:
                    analysis["security_critical"].append({
                        "name": name,
                        "version": version,
                        "reason": "Security-critical package"
                    })
            
            print(f"  ✅ Analyzed {analysis['total_packages']} packages")
            return analysis
            
        except Exception as e:
            print(f"  💥 Dependency analysis error: {e}")
            return {"error": str(e)}
    
    def _is_top_level_dependency(self, package_name: str) -> bool:
        """Check if package is a top-level dependency."""
        # Check in pyproject.toml
        pyproject_file = self.project_root / "pyproject.toml"
        if pyproject_file.exists():
            with open(pyproject_file, 'r') as f:
                content = f.read()
                return package_name.lower() in content.lower()
        
        # Check in requirements.txt
        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            with open(requirements_file, 'r') as f:
                content = f.read()
                return package_name.lower() in content.lower()
        
        return False
    
    def _perform_security_analysis(self, sbom_files: Dict[str, str]) -> Dict[str, Any]:
        """Perform security analysis on SBOM data."""
        print("🔒 Performing security analysis...")
        
        security_summary = {
            "vulnerabilities_found": 0,
            "critical_vulns": 0,
            "high_vulns": 0,
            "medium_vulns": 0,
            "low_vulns": 0,
            "vulnerable_packages": [],
            "recommendations": []
        }
        
        # Use Grype for vulnerability scanning if available
        if "json" in sbom_files:
            try:
                grype_output_file = self.sbom_dir / "vulnerabilities.json"
                
                grype_cmd = [
                    "grype",
                    "sbom:" + sbom_files["json"],
                    "-o", "json",
                    "--file", str(grype_output_file)
                ]
                
                result = subprocess.run(
                    grype_cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0 and grype_output_file.exists():
                    with open(grype_output_file, 'r') as f:
                        vuln_data = json.load(f)
                    
                    matches = vuln_data.get("matches", [])
                    security_summary["vulnerabilities_found"] = len(matches)
                    
                    # Categorize by severity
                    for match in matches:
                        severity = match.get("vulnerability", {}).get("severity", "unknown").lower()
                        
                        if severity == "critical":
                            security_summary["critical_vulns"] += 1
                        elif severity == "high":
                            security_summary["high_vulns"] += 1
                        elif severity == "medium":
                            security_summary["medium_vulns"] += 1
                        elif severity == "low":
                            security_summary["low_vulns"] += 1
                        
                        # Track vulnerable packages
                        artifact = match.get("artifact", {})
                        vuln_info = match.get("vulnerability", {})
                        
                        security_summary["vulnerable_packages"].append({
                            "name": artifact.get("name", "unknown"),
                            "version": artifact.get("version", "unknown"),
                            "vulnerability_id": vuln_info.get("id", "unknown"),
                            "severity": severity,
                            "description": vuln_info.get("description", ""),
                            "fixed_in": vuln_info.get("fix", {}).get("versions", [])
                        })
                    
                    print(f"  ✅ Found {security_summary['vulnerabilities_found']} vulnerabilities")
                    
                    # Generate recommendations
                    if security_summary["critical_vulns"] > 0:
                        security_summary["recommendations"].append(
                            "CRITICAL: Address critical vulnerabilities immediately"
                        )
                    
                    if security_summary["high_vulns"] > 0:
                        security_summary["recommendations"].append(
                            "HIGH: Update packages with high-severity vulnerabilities"
                        )
                    
                    if security_summary["vulnerabilities_found"] == 0:
                        security_summary["recommendations"].append(
                            "Good security posture: No known vulnerabilities found"
                        )
                
            except subprocess.TimeoutExpired:
                print("  ⏰ Security analysis timed out")
                security_summary["error"] = "Analysis timed out"
            except Exception as e:
                print(f"  ⚠️ Security analysis not available: {e}")
                security_summary["error"] = str(e)
        
        return security_summary
    
    def _assess_supply_chain_risks(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Assess supply chain risks based on dependency analysis."""
        print("⚠️ Assessing supply chain risks...")
        
        risks = []
        
        # Risk 1: Too many dependencies
        total_packages = analysis.get("total_packages", 0)
        if total_packages > 100:
            risks.append({
                "type": "dependency_bloat",
                "severity": "medium",
                "description": f"High number of dependencies ({total_packages})",
                "recommendation": "Review and minimize dependencies where possible",
                "impact": "Increased attack surface and maintenance burden"
            })
        
        # Risk 2: Unmaintained packages (simplified check)
        versions = analysis.get("versions", {})
        potentially_outdated = []
        
        for pkg_name, version in versions.items():
            # Simple heuristic: very old version patterns
            if any(old_pattern in version for old_pattern in ["0.1.", "0.2.", "1.0.", "1.1."]):
                potentially_outdated.append(pkg_name)
        
        if potentially_outdated:
            risks.append({
                "type": "outdated_dependencies",
                "severity": "medium",
                "description": f"Potentially outdated packages: {', '.join(potentially_outdated[:5])}",
                "recommendation": "Review and update old package versions",
                "impact": "Security vulnerabilities and compatibility issues"
            })
        
        # Risk 3: License compliance
        licenses = analysis.get("licenses", {})
        restrictive_licenses = [
            license_name for license_name in licenses.keys()
            if any(restrictive in license_name.lower() for restrictive in ["gpl", "agpl", "commercial"])
        ]
        
        if restrictive_licenses:
            risks.append({
                "type": "license_compliance",
                "severity": "high",
                "description": f"Restrictive licenses detected: {', '.join(restrictive_licenses)}",
                "recommendation": "Review license compatibility for commercial use",
                "impact": "Legal and compliance risks"
            })
        
        # Risk 4: Critical package dependencies
        security_critical = analysis.get("security_critical", [])
        if len(security_critical) > 5:
            risks.append({
                "type": "security_critical_deps",
                "severity": "medium",
                "description": f"Many security-critical dependencies ({len(security_critical)})",
                "recommendation": "Ensure security-critical packages are regularly updated",
                "impact": "High impact if vulnerabilities are found"
            })
        
        print(f"  ✅ Identified {len(risks)} supply chain risks")
        return risks
    
    def _generate_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive SBOM and security report."""
        report_file = self.sbom_dir / "sbom_comprehensive_report.json"
        
        # Add metadata
        results["metadata"] = {
            "generator": "Advanced SBOM Generator",
            "version": "1.0.0",
            "project_root": str(self.project_root),
            "sbom_directory": str(self.sbom_dir),
            "syft_version": self._get_syft_version(),
            "generation_duration": "calculated_at_runtime"
        }
        
        # Save comprehensive JSON report
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate markdown summary
        markdown_file = self.sbom_dir / "sbom_summary.md"
        self._generate_markdown_report(markdown_file, results)
        
        print(f"📄 Comprehensive report generated: {report_file}")
        return str(report_file)
    
    def _get_syft_version(self) -> str:
        """Get Syft version for metadata."""
        try:
            result = subprocess.run(
                ["syft", "version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except Exception:
            pass
        return "unknown"
    
    def _generate_markdown_report(self, file_path: Path, results: Dict[str, Any]):
        """Generate markdown SBOM summary report."""
        with open(file_path, 'w') as f:
            f.write(f"# SBOM Report: {results['project_name']}\n\n")
            f.write(f"**Generated:** {results['timestamp']}\n\n")
            
            # Summary
            analysis = results.get('analysis', {})
            f.write("## Summary\n\n")
            f.write(f"- **Total Packages:** {analysis.get('total_packages', 'N/A')}\n")
            f.write(f"- **SBOM Formats Generated:** {', '.join(results['formats_generated'])}\n")
            
            security = results.get('security_summary', {})
            f.write(f"- **Vulnerabilities Found:** {security.get('vulnerabilities_found', 'N/A')}\n")
            f.write(f"- **Supply Chain Risks:** {len(results.get('supply_chain_risks', []))}\n\n")
            
            # Package Analysis
            if analysis:
                f.write("## Package Analysis\n\n")
                
                # By ecosystem
                by_ecosystem = analysis.get('by_ecosystem', {})
                if by_ecosystem:
                    f.write("### By Ecosystem\n\n")
                    for ecosystem, count in by_ecosystem.items():
                        f.write(f"- **{ecosystem}:** {count} packages\n")
                    f.write("\n")
                
                # Licenses
                licenses = analysis.get('licenses', {})
                if licenses:
                    f.write("### License Distribution\n\n")
                    for license_name, count in list(licenses.items())[:10]:  # Top 10
                        f.write(f"- **{license_name}:** {count} packages\n")
                    f.write("\n")
            
            # Security Summary
            if security and security.get('vulnerabilities_found', 0) > 0:
                f.write("## Security Analysis\n\n")
                f.write(f"- **Critical:** {security.get('critical_vulns', 0)}\n")
                f.write(f"- **High:** {security.get('high_vulns', 0)}\n")
                f.write(f"- **Medium:** {security.get('medium_vulns', 0)}\n")
                f.write(f"- **Low:** {security.get('low_vulns', 0)}\n\n")
                
                # Top vulnerable packages
                vulnerable = security.get('vulnerable_packages', [])[:5]
                if vulnerable:
                    f.write("### Top Vulnerable Packages\n\n")
                    for pkg in vulnerable:
                        f.write(f"- **{pkg['name']} {pkg['version']}**: {pkg['vulnerability_id']} ({pkg['severity']})\n")
                    f.write("\n")
            
            # Supply Chain Risks
            risks = results.get('supply_chain_risks', [])
            if risks:
                f.write("## Supply Chain Risks\n\n")
                for risk in risks:
                    f.write(f"### {risk['type'].replace('_', ' ').title()} ({risk['severity'].upper()})\n\n")
                    f.write(f"**Description:** {risk['description']}\n\n")
                    f.write(f"**Recommendation:** {risk['recommendation']}\n\n")
                    f.write(f"**Impact:** {risk['impact']}\n\n")
            
            # Recommendations
            all_recommendations = []
            all_recommendations.extend(security.get('recommendations', []))
            for risk in risks:
                all_recommendations.append(risk['recommendation'])
            
            if all_recommendations:
                f.write("## Recommendations\n\n")
                for i, rec in enumerate(all_recommendations, 1):
                    f.write(f"{i}. {rec}\n")


def main():
    """Main entry point for SBOM generation."""
    parser = argparse.ArgumentParser(description="Advanced SBOM Generator")
    parser.add_argument("--project-name", help="Project name for SBOM")
    parser.add_argument("--include-dev", action="store_true", default=True,
                       help="Include development dependencies")
    parser.add_argument("--exclude-dev", action="store_true",
                       help="Exclude development dependencies")
    parser.add_argument("--include-container", action="store_true", default=True,
                       help="Generate container SBOM if Dockerfile exists")
    parser.add_argument("--format", choices=["json", "spdx-json", "cyclonedx-json", "table"],
                       help="Generate only specific format")
    parser.add_argument("--output-dir", help="Output directory for SBOM files")
    
    args = parser.parse_args()
    
    # Create generator
    project_name = args.project_name or "fair-credit-scorer-bias-mitigation"
    generator = SBOMGenerator(project_name)
    
    # Override output directory if specified
    if args.output_dir:
        generator.sbom_dir = Path(args.output_dir)
        generator.sbom_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine dev dependencies inclusion
    include_dev = args.include_dev and not args.exclude_dev
    
    try:
        # Generate SBOM
        result = generator.generate_comprehensive_sbom(
            include_dev_deps=include_dev,
            include_container=args.include_container
        )
        
        print(f"\n🎉 SBOM generation completed successfully!")
        print(f"   Project: {result['project_name']}")
        print(f"   Formats: {', '.join(result['formats_generated'])}")
        print(f"   Packages: {result['analysis'].get('total_packages', 'N/A')}")
        print(f"   Vulnerabilities: {result['security_summary'].get('vulnerabilities_found', 'N/A')}")
        print(f"   Report: {result['report_path']}")
        
        # Exit with error if critical vulnerabilities found
        critical_vulns = result['security_summary'].get('critical_vulns', 0)
        if critical_vulns > 0:
            print(f"\n⚠️ WARNING: {critical_vulns} critical vulnerabilities found!")
            sys.exit(1)
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n⚠️ SBOM generation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 SBOM generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()