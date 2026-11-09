BEGIN {
	phase = "analysis";
	ok = 1;

	expect_tables = 2;
	table[1] = "Stats by (Port, Lane):";
	table[2] = "Stats by (Vendor, Device):";

	#
	# Make sure a complete set of lanes appears in the output.  This list is
	# likely model specific.
	#
	want_lanes["M.2 East(A)"] = 4;
	want_lanes["M.2 East(A) bridge"] = 4;
	want_lanes["M.2 West(B)"] = 4;
	want_lanes["M.2 West(B) bridge"] = 4;
	want_lanes["N0(A) bridge"] = 4;
	want_lanes["N0(A)"] = 4;
	want_lanes["N1(B) bridge"] = 4;
	want_lanes["N1(B)"] = 4;
	want_lanes["N2(C) bridge"] = 4;
	want_lanes["N2(C)"] = 4;
	want_lanes["N3(D) bridge"] = 4;
	want_lanes["N3(D)"] = 4;
	want_lanes["N4(E) bridge"] = 4;
	want_lanes["N4(E)"] = 4;
	want_lanes["N5(F) bridge"] = 4;
	want_lanes["N5(F)"] = 4;
	want_lanes["N6(G) bridge"] = 4;
	want_lanes["N6(G)"] = 4;
	want_lanes["N7(H) bridge"] = 4;
	want_lanes["N7(H)"] = 4;
	want_lanes["N8(I) bridge"] = 4;
	want_lanes["N8(I)"] = 4;
	want_lanes["N9(J) bridge"] = 4;
	want_lanes["N9(J)"] = 4;

	tables = 0;
	pass_count = 0;
}

phase == "analysis" && NF == 0 {
	#
	# The analysis parameters section ends with a blank line.
	#
	phase = "wait_for_table";
	printf("\n");
	next;
}

phase == "analysis" {
	#
	# Pass the analysis parameters straight through.
	#
	print;
}

phase == "wait_for_table" && NF != 0 {
	tables++;
	if ($0 != table[tables]) {
		printf("unexpected heading: %s\n", $0);
		ok = 0;
		exit(1);
	}

	header[tables] = $0 "\n";
	body[tables] = "";
	pass_offset[tables] = 0;
	port_offset[tables] = 0;
	lane_offset[tables] = 0;

	phase = "table_header";
	next;
}

phase == "table_header" {
	header[tables] = header[tables] $0 "\n";
}

phase == "table_header" && $NF == "PASS" {
	pass_offset[tables] = index($0, "PASS");
}

phase == "table_header" && $1 == /Port/ {
	port_offset[tables] = index($0, "Port");
}

phase == "table_header" && /Lane/ {
	lane_offset[tables] = index($0, "Lane");
}

phase == "table_header" && /----/ {
	if (pass_offset[tables] == 0) {
		printf("no PASS column?\n");
		ok = 0;
		exit(1);
	}

	if (tables == 1 && (lane_offset[tables] <= port_offset[tables])) {
		printf("weird Port and Lane columns?\n");
		ok = 0;
		exit(1);
	}

	phase = "table_body";
	next;
}

phase == "table_body" && NF == 0 {
	#
	# The first table ends with a blank line.
	#
	phase = "wait_for_table";
	next;
}

#
# The first table lists ports, and we want to make sure we get a complete set in
# the results.
#
phase == "table_body" && tables == 1 {
	port = substr($0, port_offset[tables],
	     lane_offset[tables] - port_offset[tables] - 1);
	gsub(" *$", "", port);

	lanes[port] += 1;
}

phase == "table_body" {
	res = substr($0, pass_offset[tables], 1);
	if (res == "Y") {
		pass_count++;
	} else if (res == "N") {
		body[tables] = body[tables] $0 "\n";
		fail_count[tables]++;
	} else {
		printf("strange body row: %s\n", $0);
		ok = 0;
		exit(1);
	}
}

END {
	if (ok != 1) {
		printf("ERROR: results file form not as expected\n");
		exit(1);
	}

	if (phase != "table_body" || tables != expect_tables) {
		printf("ERROR: results file was incomplete!\n");
	}

	for (p in want_lanes) {
		if (lanes[p] != want_lanes[p]) {
			printf("ERROR: port \"%s\" had %d lanes, not %d\n", p,
			    lanes[p], want_lanes[p]);
			ok = 0;
		}
	}

	for (l in lanes) {
		if (want_lanes[l] < 1) {
			printf("ERROR: extraneous lane \"%s\"\n", l);
			ok = 0;
		}
	}

	total_fails = 0;
	for (i = 1; i <= expect_tables; i++) {
		total_fails += fail_count[i];
		if (fail_count[i] != 0) {
			printf("%s%s\n", header[i], body[i]);
			ok = 0;
		}
	}

	if (ok != 1) {
		if (total_fails != 0) {
			printf("ERROR: %d margining failures detected!\n",
			    total_fails);
		}
		exit(1);
	}

	printf(" * margining results ok!\n");
	exit(0);
}
