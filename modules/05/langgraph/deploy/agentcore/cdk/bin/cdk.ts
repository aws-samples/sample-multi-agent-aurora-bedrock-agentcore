#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import * as fs from 'fs';
import * as path from 'path';
import { AgentCoreStack } from '../lib/cdk-stack';

const app = new cdk.App();

// Read project spec from agentcore.json (three levels up from dist/bin/)
const projectRoot = path.resolve(__dirname, '..', '..', '..');
const specPath = path.join(projectRoot, 'agentcore.json');
const targetsPath = path.join(projectRoot, 'aws-targets.json');

if (!fs.existsSync(specPath)) {
  throw new Error(`agentcore.json not found at ${specPath}`);
}

const spec = JSON.parse(fs.readFileSync(specPath, 'utf-8'));
const targets = JSON.parse(fs.readFileSync(targetsPath, 'utf-8'));

// Deploy to each target
for (const target of targets) {
  const stackName = `AgentCore-${spec.name}-${target.name}`;

  new AgentCoreStack(app, stackName, {
    spec,
    env: {
      account: target.account,
      region: target.region,
    },
  });
}
