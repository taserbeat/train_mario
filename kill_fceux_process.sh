#!/bin/bash

ps aux | grep fceux.app | grep -v grep | awk '{ print "kill -9", $2 }' | sh