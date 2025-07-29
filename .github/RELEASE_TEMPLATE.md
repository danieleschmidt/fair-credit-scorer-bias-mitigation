# Release Template

Use this template when creating releases for the fair credit scorer project.

## Pre-Release Checklist

- [ ] All tests passing in CI
- [ ] Documentation updated
- [ ] CHANGELOG.md updated with new version
- [ ] Version bumped in pyproject.toml
- [ ] Security scan completed with no critical issues
- [ ] Performance benchmarks within acceptable thresholds
- [ ] All dependencies up to date

## Release Process

### Automated Release (Recommended)

```bash
# Prepare patch release (bug fixes)
./scripts/release_automation.py prepare --bump patch

# Prepare minor release (new features)
./scripts/release_automation.py prepare --bump minor

# Prepare major release (breaking changes)
./scripts/release_automation.py prepare --bump major

# Full release with tag and build
./scripts/release_automation.py full --bump patch
```

### Manual Release Steps

1. **Update Version**
   ```bash
   # Edit pyproject.toml version field
   vim pyproject.toml
   ```

2. **Update Changelog**
   ```bash
   # Add new version section to CHANGELOG.md
   vim CHANGELOG.md
   ```

3. **Create Tag**
   ```bash
   git tag -a v0.2.1 -m "Release version 0.2.1"
   git push origin v0.2.1
   ```

4. **Build Package**
   ```bash
   python -m build
   ```

## Release Types

### Patch Release (0.0.X)
- Bug fixes
- Security updates
- Documentation improvements
- No breaking changes

### Minor Release (0.X.0)
- New features
- Deprecations (with backward compatibility)
- Performance improvements
- Enhanced functionality

### Major Release (X.0.0)
- Breaking changes
- Major architectural changes
- Removal of deprecated features
- API changes

## Post-Release Actions

- [ ] GitHub release created with release notes
- [ ] Package published to PyPI (if applicable)
- [ ] Documentation site updated
- [ ] Social media announcement (if applicable)
- [ ] Team notification sent
- [ ] Next milestone planning updated

## Release Notes Template

```markdown
# Release X.Y.Z

## 🎉 What's New
- List new features and enhancements

## 🐛 Bug Fixes
- List bug fixes and corrections

## 🔒 Security
- List security improvements

## 📖 Documentation
- List documentation updates

## 🔧 Maintenance
- List maintenance and internal improvements

## 📋 Full Changelog
**Full Changelog**: https://github.com/danieleschmidt/fair-credit-scorer-bias-mitigation/compare/vX.Y.Z-1...vX.Y.Z
```

## Emergency Release Process

For critical security fixes or urgent bug fixes:

1. **Create hotfix branch**
   ```bash
   git checkout -b hotfix/critical-fix main
   ```

2. **Make minimal fix**
   - Keep changes as small as possible
   - Focus only on the critical issue

3. **Fast-track testing**
   - Run relevant tests
   - Skip non-critical validations if needed

4. **Emergency release**
   ```bash
   ./scripts/release_automation.py full --bump patch --changes "Critical security fix"
   ```

5. **Post-release notification**
   - Notify all users immediately
   - Document the issue and fix
   - Schedule follow-up review

## Quality Gates

All releases must pass these quality gates:

- ✅ All automated tests pass
- ✅ Security scan shows no critical vulnerabilities  
- ✅ Performance benchmarks within ±20% of baseline
- ✅ Code coverage ≥80%
- ✅ Documentation build succeeds
- ✅ Package builds successfully
- ✅ CHANGELOG.md updated
- ✅ Version consistency across files

## Rollback Process

If issues are discovered post-release:

1. **Immediate action**
   ```bash
   # Revert problematic changes
   git revert <commit-hash>
   
   # Create emergency patch
   ./scripts/release_automation.py full --bump patch --changes "Rollback: revert problematic changes"
   ```

2. **Communication**
   - Update GitHub release with warning
   - Notify users through available channels
   - Document issue and resolution

3. **Analysis**
   - Conduct post-mortem
   - Update processes to prevent recurrence
   - Enhance testing if needed

## Support

For questions about the release process:
- Check [CONTRIBUTING.md](../CONTRIBUTING.md)
- Create GitHub issue with `release` label  
- Contact maintainers through established channels